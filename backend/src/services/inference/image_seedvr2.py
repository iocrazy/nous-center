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

    PR-1 骨架:只建结构 + 验证 vendored import 可用。load()/upscale() 在 PR-2 实现
    (调 seedvr2_vendor.core.generation_utils.prepare_runner + generation_phases 4 阶段)。
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

    @staticmethod
    def vendored_import_ok() -> bool:
        """PR-1 自检:vendored 推理核心能 import(兼容 patch 生效)。"""
        from src.services.inference.seedvr2_vendor.core import (  # noqa: PLC0415
            generation_phases,
            generation_utils,
        )
        return hasattr(generation_utils, "prepare_runner") and hasattr(
            generation_phases, "upscale_all_batches"
        )
