"""SeedVR2 超分上采样引擎 adapter —— 接 NumZ vendored 推理核心到 nous-center。

SeedVR2 = ByteDance 的 one-step diffusion 超分上采样器(DiT 7B + 专用 video VAE)。
推理核心 vendored 在 `seedvr2_vendor/`(NumZ src,Apache 2.0,不重写),本文件是**桥** ——
把 nous-center 的 InferenceAdapter 接口转成 NumZ 的 prepare_runner + 4 阶段
(encode → DiT one-step upscale → decode → post)。对标 ComfyUI 的 interfaces/ 桥层。

PR-3a:符合 InferenceAdapter ABC —— `__init__(paths, device, **params)` /
`async load(device)` / `async infer(UpscaleRequest)→InferenceResult`,让 ModelManager
能跟管 Flux2/anima 一样管 SeedVR2(LRU/显存/跨进程可见)。内核仍是 PR-2 的同步
`upscale(PIL)→PIL`(忠实复刻 NumZ CLI 的 4 阶段串法),infer 只是 decode 图 +
to_thread 包一层。project_seedvr2_pr3_design。
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, ClassVar

# **必须在 import 任何 seedvr2_vendor 模块前** patch transformers 5.6-dev 的 flash_attn bug。
from src.services.inference.seedvr2_compat import apply_seedvr2_compat_patches

apply_seedvr2_compat_patches()

from src.services.inference.base import (  # noqa: E402 — 必须在 compat patch 之后
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UpscaleRequest,
    UsageMeter,
)

# 默认模型(CLI 白名单认的 7B fp8;HF 自动下到 model_dir)。
DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"


def _decode_image(src: str) -> Any:
    """req.image(base64 data URI 或本地路径)→ PIL.Image(RGB)。"""
    from PIL import Image  # noqa: PLC0415

    if src.startswith("data:"):
        # "data:image/png;base64,...." → 取逗号后的 base64 体
        _, _, b64 = src.partition(",")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(src).convert("RGB")


class SeedVR2UpscaleBackend(InferenceAdapter):
    """SeedVR2 超分引擎(InferenceAdapter)。输入图 + 目标分辨率 → 超分图。

    paths 接口(由 ModelManager._get_or_load_seedvr2_adapter 装填):
      paths["model_dir"] = SEEDVR2 模型目录(DiT/VAE 所在;缺模型时 HF 自动下到此)
      paths["dit"]       = DiT 文件名(可选,默认 DEFAULT_DIT;NumZ 白名单语义,非全路径)
      paths["vae"]       = VAE 文件名(可选,默认 DEFAULT_VAE)

    内核(load 装 runner;upscale 跑 NumZ 4 阶段)忠实复刻 NumZ CLI 的 process_single_file
    串法(setup_generation_context → prepare_runner → encode/upscale/decode/postprocess
    → ctx['final_video'])。
    """

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    # 7B fp8 DiT + video VAE;Pro 6000 上跑绰绰有余,给保守上界供 VRAM 守卫/估算用。
    estimated_vram_mb: ClassVar[int] = 24000

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any) -> None:
        super().__init__(paths, device, **params)
        self.model_dir = paths.get("model_dir") or params.get("model_dir")
        if not self.model_dir:
            raise RuntimeError("SeedVR2UpscaleBackend 需要 paths['model_dir'](SEEDVR2 模型目录)")
        self.dit_model = paths.get("dit") or params.get("dit_model") or DEFAULT_DIT
        self.vae_model = paths.get("vae") or params.get("vae_model") or DEFAULT_VAE
        self._runner: Any | None = None
        self._ctx_base: dict[str, Any] | None = None
        self._debug: Any | None = None

    @staticmethod
    def vendored_import_ok() -> bool:
        """PR-1 自检:vendored 推理核心能 import(兼容 patch 生效)。"""
        from src.services.inference.seedvr2_vendor.src.core import (  # noqa: PLC0415
            generation_phases,
            generation_utils,
        )
        return hasattr(generation_utils, "prepare_runner") and hasattr(
            generation_phases, "upscale_all_batches"
        )

    async def load(self, device: str) -> None:
        """ABC 入口:重 GPU 装载丢 to_thread,不阻塞 runner 事件循环(否则 supervisor
        ping 超时误判 crash,见 image_anima 同款处理)。"""
        self.device = device
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        """装 DiT + VAE runner(NumZ prepare_runner)+ 备好 ctx(cache_context + text_embeds)。
        缺模型时 prepare_runner 从 HF 自动下到 model_dir。对齐 NumZ CLI process_single_file。"""
        from src.services.inference.seedvr2_vendor.src.core.generation_utils import (  # noqa: PLC0415
            load_text_embeddings,
            prepare_runner,
            setup_generation_context,
        )
        from src.services.inference.seedvr2_vendor.src.utils.constants import (  # noqa: PLC0415
            get_script_directory,
        )
        from src.services.inference.seedvr2_vendor.src.utils.debug import Debug  # noqa: PLC0415

        self._debug = Debug(enabled=False)
        ctx = setup_generation_context(
            dit_device=self.device,
            vae_device=self.device,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            debug=self._debug,
        )
        # prepare_runner 改/装 ctx in-place,返回 (runner, cache_context)。
        self._runner, cache_context = prepare_runner(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            model_dir=self.model_dir,
            debug=self._debug,
            ctx=ctx,
            attention_mode="sdpa",  # flash_attn 装不上,SDPA 回退(SeedVR2 支持)
        )
        # NumZ CLI 在 prepare_runner 后、encode 前手动补的两步(否则 encode 撞 KeyError):
        ctx["cache_context"] = cache_context
        ctx["text_embeds"] = load_text_embeddings(
            get_script_directory(), ctx["dit_device"], ctx["compute_dtype"], self._debug,
        )
        self._ctx_base = ctx
        self._model = self._runner  # ABC is_loaded → True

    def unload(self) -> None:
        """释放 DiT+VAE runner + 显存。base.unload 只置 _model=None,但 SeedVR2 还持
        _runner(DiT 7B + video VAE)+ ctx,不显式拆 + empty_cache 显存不降(同 anima/modular)。"""
        self._runner = None
        self._ctx_base = None
        self._model = None
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort
            pass

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Any | None = None,
        cancel_flag: Any | None = None,
    ) -> InferenceResult:
        """UpscaleRequest → InferenceResult(image/png)。decode 输入图 → to_thread 跑
        同步 upscale 内核(CUDA 阻塞,不能占 runner 事件循环)→ PNG bytes。

        progress_callback / cancel_flag 暂未细化(SeedVR2 是 one-step,无 per-step 进度;
        PR-3b runner 接入时按需桥接 stage 级进度)。
        """
        if not isinstance(req, UpscaleRequest):
            raise TypeError(f"SeedVR2UpscaleBackend 只接受 UpscaleRequest,收到 {type(req).__name__}")

        pil = _decode_image(req.image)
        seed = req.seed if req.seed is not None else 42

        def _run() -> Any:
            return self.upscale(
                pil,
                resolution=req.resolution,
                seed=seed,
                color_correction=req.color_correction,
                latent_noise_scale=req.latent_noise_scale,
                input_noise_scale=req.input_noise_scale,
            )

        t = time.monotonic()
        out = await asyncio.to_thread(_run)
        latency_ms = int((time.monotonic() - t) * 1000)

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return InferenceResult(
            media_type="image/png",
            data=buf.getvalue(),
            metadata={
                "width": out.width,
                "height": out.height,
                "resolution": req.resolution,
                "seed": seed,
                "engine": "seedvr2",
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )

    def upscale(
        self,
        image: "Any",  # noqa: UP037 — PIL.Image.Image,惰性 import 避免顶层依赖
        resolution: int = 1080,
        seed: int = 42,
        batch_size: int = 1,
        color_correction: str = "lab",
        latent_noise_scale: float = 0.0,
        input_noise_scale: float = 0.0,
    ) -> "Any":
        """单图超分。image(PIL.Image)→ 超分后 PIL.Image。忠实复刻 NumZ 4 阶段。

        resolution = 目标短边(SeedVR2 语义:输出最短边像素;非倍数)。
        """
        if self._runner is None:
            raise RuntimeError("SeedVR2 未 load")
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from src.services.inference.seedvr2_vendor.src.core.generation_phases import (  # noqa: PLC0415
            decode_all_batches,
            encode_all_batches,
            postprocess_all_batches,
            upscale_all_batches,
        )

        # 输入图 → frames_tensor [1,H,W,C] float16 [0,1] RGB(对齐 NumZ extract_frames_from_image)。
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        frames = torch.from_numpy(rgb[None, ...]).to(torch.float16)

        # ctx 每次 upscale 重置(NumZ ctx 阶段间 in-place 累积,复用 runner 但 ctx 要新)。
        ctx = dict(self._ctx_base) if self._ctx_base else {}

        ctx = encode_all_batches(
            self._runner, ctx=ctx, images=frames, debug=self._debug,
            batch_size=batch_size, uniform_batch_size=False, seed=seed,
            progress_callback=None, temporal_overlap=0,
            resolution=resolution, max_resolution=0,
            input_noise_scale=input_noise_scale, color_correction=color_correction,
        )
        ctx = upscale_all_batches(
            self._runner, ctx=ctx, debug=self._debug, progress_callback=None,
            seed=seed, latent_noise_scale=latent_noise_scale, cache_model=False,
        )
        ctx = decode_all_batches(
            self._runner, ctx=ctx, debug=self._debug, progress_callback=None,
            cache_model=False,
        )
        ctx = postprocess_all_batches(
            ctx=ctx, debug=self._debug, progress_callback=None,
            color_correction=color_correction, prepend_frames=0,
            temporal_overlap=0, batch_size=batch_size,
        )

        out = ctx["final_video"]  # [N,H,W,C] [0,1]
        if out.is_cuda:
            out = out.cpu()
        if out.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
            out = out.to(torch.float32)
        arr = (out[0].clamp(0, 1).float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
