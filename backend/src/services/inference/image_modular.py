"""Modular Diffusers 图像引擎(ModularPipeline + ComponentsManager)。

spec: 2026-05-22-image-engine-modular-diffusers-design.md。PR-1:与现有
`DiffusersImageBackend`(自写 ImageSampler)**并存灰度**,经 `NOUS_IMAGE_ENGINE`
选择(默认 legacy)。PR-4 才删旧。

约束(plan-eng-review D2):`diffusers.modular*` / `diffusers` 的 import **只允许经
`_import_modular()` 这一个函数**(blast-radius 隔离 + conftest mock torch 时模块顶层
不 import diffusers/torch,避免 collection 崩)。
"""
from __future__ import annotations

import io
import time
from typing import Any, ClassVar

from src.services.inference.base import (
    ImageRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


def _import_modular() -> tuple[Any, Any]:
    """Lazy import —— diffusers 的 Modular API 只在这里取(D2 隔离)。

    测试可 monkeypatch 本函数返回 fake,从而无需真 diffusers/GPU 验证参数映射。
    """
    from diffusers import ComponentsManager, ModularPipeline  # noqa: PLC0415

    return ModularPipeline, ComponentsManager


def _torch_dtype(dtype_str: str) -> Any:
    """'bfloat16'/'float16'/'float32' → torch.dtype;'default' → None(原生精度)。"""
    import torch  # noqa: PLC0415

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "default": None,
    }.get(dtype_str, torch.bfloat16)


class ModularImageBackend(InferenceAdapter):
    """ModularPipeline 后端,实现现有 `InferenceAdapter.infer` 接口(与 TTS/LLM 一致)。

    PR-1 只做 HF-layout 基线(bf16);量化桥接 PR-2、LoRA PR-3。
    """

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    estimated_vram_mb: ClassVar[int] = 0

    def __init__(
        self,
        repo: str,
        device: str = "cuda",
        *,
        dtype: str = "bfloat16",
        components_manager: Any = None,
        **params: Any,
    ):
        super().__init__(paths={"main": repo}, device=device, **params)
        self.repo = repo
        self.dtype = dtype
        self._cm = components_manager
        self._pipe: Any = None

    async def load(self, device: str) -> None:
        """对齐 ABC;实际 pipeline 构建在首次 infer 时 lazy(_ensure_pipe)。"""
        self.device = device

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        modular_pipeline_cls, components_manager_cls = _import_modular()
        cm = self._cm or components_manager_cls()
        pipe = modular_pipeline_cls.from_pretrained(self.repo, components_manager=cm)
        pipe.load_components(torch_dtype=_torch_dtype(self.dtype))
        pipe.to(self.device)
        self._pipe = pipe
        self._model = pipe  # is_loaded → True
        return pipe

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        if not isinstance(req, ImageRequest):
            raise TypeError(f"ModularImageBackend 只接受 ImageRequest,收到 {type(req).__name__}")
        import torch  # noqa: PLC0415

        pipe = self._ensure_pipe()
        gen = torch.Generator(device=self.device)
        if req.seed is not None:
            gen = gen.manual_seed(req.seed)

        t = time.monotonic()
        # 注:Flux2-klein 是 guidance-distilled,spike 未传 cfg;PR-1 基线沿用,
        # cfg_scale 映射(guidance_scale)留 PR-3 随多 CLIP/参数面一起接。
        out = pipe(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            generator=gen,
        )
        latency_ms = int((time.monotonic() - t) * 1000)

        img = out.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return InferenceResult(
            media_type="image/png",
            data=buf.getvalue(),
            metadata={
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "engine": "modular",
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
