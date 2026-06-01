"""SeedVR2 超分上采样引擎 adapter —— 接 NumZ vendored 推理核心到 nous-center。

SeedVR2 = ByteDance 的 one-step diffusion 超分上采样器(DiT 7B + 专用 video VAE)。
推理核心 vendored 在 `seedvr2_vendor/`(NumZ src,Apache 2.0,不重写),本文件是**桥** ——
把 nous-center 的 InferenceAdapter 接口转成 NumZ 的 prepare_runner + 4 阶段
(encode → DiT one-step upscale → decode → post)。对标 ComfyUI 的 interfaces/ 桥层。

PR-1(本文件):骨架 + import wiring + 兼容 patch。真推理(模型加载 + 4 阶段)在 PR-2。
project_seedvr2_integration。
"""
from __future__ import annotations

from typing import Any

# **必须在 import 任何 seedvr2_vendor 模块前** patch transformers 5.6-dev 的 flash_attn bug。
from src.services.inference.seedvr2_compat import apply_seedvr2_compat_patches

apply_seedvr2_compat_patches()

# 默认模型(CLI 白名单认的 7B fp8;HF 自动下到 model_dir)。
DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
DEFAULT_VAE = "ema_vae_fp16.safetensors"


class SeedVR2UpscaleBackend:
    """SeedVR2 超分引擎。输入图 + 目标分辨率 → 超分图。

    PR-2:真推理(load 装 runner;upscale 跑 NumZ 4 阶段)。忠实复刻 NumZ CLI 的
    process_single_file 串法(setup_generation_context → prepare_runner →
    encode/upscale/decode/postprocess → ctx['final_video'])。
    """

    def __init__(
        self,
        model_dir: str,
        dit_model: str = DEFAULT_DIT,
        vae_model: str = DEFAULT_VAE,
        device: str = "cuda",
    ) -> None:
        self.model_dir = model_dir
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.device = device
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

    def load(self) -> None:
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

    @property
    def is_loaded(self) -> bool:
        return self._runner is not None

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
