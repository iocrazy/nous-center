"""AnimaImageBackend — nous InferenceAdapter wrapping arch_anima.AnimaPipeline。

接 PR-anima-6 engine 集成。让 nous model_manager + runner 能跟管 Flux2 一样管 anima:
  - get_or_load_image_adapter(components, pipeline_class="AnimaPipeline") → AnimaImageBackend
  - adapter.infer(ImageRequest) → InferenceResult(media_type=image/png + PIL bytes + meta)

跟 `image_modular.ModularImageBackend` 接口对齐(InferenceAdapter ABC),让上游
(runner_process / workflow_executor)无须区分 anima vs Flux2。

环境变量(开发期 / PR-7 真模型路径配置;PR-anima-5 part 1 已 bundle Qwen3 config,
tokenizer 6.7M 暂不 bundle):

  NOUS_ANIMA_QWEN_TOKENIZER  qwen25_tokenizer 目录(从 ComfyUI 拷;必需)
  NOUS_ANIMA_T5_TOKENIZER    t5_tokenizer 目录(可选,启用 LLMAdapter 桥接)
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Any, ClassVar

import torch

from src.services.inference.base import (
    ImageRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


class AnimaImageBackend(InferenceAdapter):
    """Wrap arch_anima.AnimaPipeline 让 nous engine 直跑 anima。

    跟 ModularImageBackend(Flux2 用)对齐:接 paths dict / device / params,
    lazy init pipe(load() / 首次 infer() 触发);infer() 返 InferenceResult(image/png)。

    paths 接口(由 model_manager 装填):
      paths["transformer"] = anima-base-v1.0.safetensors
      paths["text_encoder"] = qwen_3_06b_base.safetensors
      paths["vae"]          = qwen_image_vae.safetensors
    qwen tokenizer 目录从 NOUS_ANIMA_QWEN_TOKENIZER env 拿(spec 决策:6.7M 不 bundle)。
    """

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    estimated_vram_mb: ClassVar[int] = 10000  # 1024 真出图 9.4 GiB(PR-7 实测)

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any) -> None:
        super().__init__(paths, device, **params)
        self.dtype = params.get("dtype", "bfloat16")
        self.pipeline_class = params.get("pipeline_class", "AnimaPipeline")
        # PR-D2 offload kwarg:本 PR 简单实现先 none/cpu/cuda:N 路径(类似 image_modular),
        # 跨卡 hook 留 PR-D2 复用(模式跟 Flux2 一致)。
        self.offload = params.get("offload", "none")
        self._pipe: Any = None

    async def load(self, device: str) -> None:
        """对齐 ABC;实际 pipeline 构建在首次 infer 时 lazy(_ensure_pipe)。"""
        self.device = device

    def unload(self) -> None:
        """释放 GPU pipeline —— 见 ModularImageBackend.unload 注释(round3 #3)。

        base.unload 只 `_model=None`,但 anima 还把 `_model=_pipe` 互引用、`_pipe`
        持 DiT+Qwen+VAE(1024 出图 9.4 GiB),不显式拆 + empty_cache 显存不降。
        """
        self._pipe = None
        self._model = None
        try:
            import gc  # noqa: PLC0415
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort
            pass

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        if self.pipeline_class != "AnimaPipeline":
            raise NotImplementedError(
                f"AnimaImageBackend 只支持 pipeline_class=AnimaPipeline,收到 {self.pipeline_class!r}",
            )

        from src.services.inference.arch_anima import AnimaPipeline  # noqa: PLC0415

        # 必需 path:三件套
        anima_w = self.paths.get("transformer")
        qwen_w = self.paths.get("text_encoder")
        vae_w = self.paths.get("vae")
        for label, p in (("transformer", anima_w), ("text_encoder", qwen_w), ("vae", vae_w)):
            if not p:
                raise RuntimeError(f"AnimaImageBackend: paths[{label!r}] missing")

        # tokenizer 目录从 env(spec 决策:6.7M 不 bundle;caller 提供)。
        qwen_tok = os.environ.get("NOUS_ANIMA_QWEN_TOKENIZER")
        if not qwen_tok:
            raise RuntimeError(
                "AnimaImageBackend 需要 NOUS_ANIMA_QWEN_TOKENIZER env(qwen25_tokenizer 目录路径);"
                "spec 2026-05-26 决策:tokenizer 6.7M 不 bundle 进 repo,caller 提供。",
            )
        t5_tok = os.environ.get("NOUS_ANIMA_T5_TOKENIZER")  # 可选

        torch_dtype = torch.bfloat16 if self.dtype in ("bfloat16", "default") else torch.float16
        self._pipe = AnimaPipeline.from_components(
            anima_weights=anima_w,
            qwen_weights=qwen_w,
            qwen_tokenizer_dir=qwen_tok,
            vae_weights=vae_w,
            t5_tokenizer_dir=t5_tok,
            device=self.device,
            dtype=torch_dtype,
        )
        self._model = self._pipe  # is_loaded → True(ABC 用)
        return self._pipe

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Any | None = None,
        cancel_flag: Any | None = None,
    ) -> InferenceResult:
        """对齐 ModularImageBackend.infer 契约 — ImageRequest → InferenceResult(image/png)。

        progress_callback:每 denoise step 发 dit_denoise 进度(+ 前后 text_encode/vae_decode
        stage),对齐 Flux2 的 ProgressTracker。cancel_flag(threading.Event):step 边界查置位 →
        raise CancelledError 中断 denoise。
        """
        if not isinstance(req, ImageRequest):
            raise TypeError(f"AnimaImageBackend 只接受 ImageRequest,收到 {type(req).__name__}")

        cfg = float(req.cfg_scale)
        neg = (getattr(req, "negative_prompt", "") or "").strip()
        total_steps = int(req.steps)

        # 进度桥接:#206 起 denoise 跑在 to_thread 工作线程 —— 回调不能直接 create_task
        # (工作线程无 running loop)。捕获事件循环,用 call_soon_threadsafe 把回调调度回 loop
        # (runner _on_progress 在 loop 上 create_task 发 NodeProgress)。runner_process.py:354
        # 正预见此「真 adapter callback 在 to_thread 工作线程」需 threadsafe 调度。
        from src.services.inference.progress_tracker import ProgressTracker  # noqa: PLC0415
        loop = asyncio.get_running_loop()

        def _emit(*a: Any, **kw: Any) -> None:
            if progress_callback is not None:
                loop.call_soon_threadsafe(lambda: progress_callback(*a, **kw))

        pt = ProgressTracker(_emit)  # throttle_ms=0 → 真 per-step
        pt.stage("text_encode", step=0, total_steps=total_steps)

        def _step(done: int) -> None:
            # denoise step 边界(工作线程):cancel 检查 + 发 dit_denoise 进度。
            if cancel_flag is not None and cancel_flag.is_set():
                raise asyncio.CancelledError()
            pt.step(done, total_steps, stage="dit_denoise")

        # _ensure_pipe(首次 ~5s 构建)+ pipe()(denoise,1024 在 3090 ~20s)是阻塞 CUDA。
        # 丢 to_thread 让 runner 事件循环空闲答 supervisor ping(否则 >10s 被误判 crash kill,
        # 见 #206 + runner_process.py:11 契约)。
        def _run() -> Any:
            pipe = self._ensure_pipe()
            return pipe(
                prompt=req.prompt,
                negative_prompt=neg,
                num_inference_steps=req.steps,
                width=req.width,
                height=req.height,
                seed=req.seed,
                guidance_scale=cfg,
                step_callback=_step,
            )

        t = time.monotonic()
        img = await asyncio.to_thread(_run)
        latency_ms = int((time.monotonic() - t) * 1000)
        pt.stage("vae_decode", progress=1.0, step=total_steps, total_steps=total_steps)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return InferenceResult(
            media_type="image/png",
            data=buf.getvalue(),
            metadata={
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "engine": "anima",
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
