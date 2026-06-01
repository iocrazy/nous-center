"""ModelArchAdapter — abstracts diffusers Pipeline family differences for ImageSampler.

PR-2 of image-component-multi-gpu spec §5.6.3. The ImageSampler's main loop is
Pipeline-family agnostic; per-arch differences (Klein has no CFG, Dev has CFG,
SDXL has dual CLIP) are isolated in adapter implementations registered by
Pipeline class name.

PR-2 ships only FluxKleinArchAdapter (matches the on-disk Flux2-Klein-9B model
verified by Task 0). Future PRs add FluxDev / SDXL / Z-Image / QwenImageEdit
adapters in ~50 LOC each.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelArchAdapter(Protocol):
    """Per-Pipeline-class settings + behavior switches consumed by ImageSampler.

    All methods are pure (no side effects) — adapter instances are singletons
    in MODEL_ARCH_REGISTRY.
    """

    def supports_cfg(self) -> bool:
        """True if this Pipeline class uses classifier-free guidance (CFG)
        in its denoise loop. Distilled models (Klein, Z-Image-Turbo) → False.
        Consumed by FluxDevArchAdapter (future PR adding CFG-branched models).
        """
        ...

    def supports_negative_prompt(self) -> bool:
        """True if encode_prompt accepts a negative_prompt argument.
        Distilled models reject it; mainline (Dev, SDXL) accept it.
        Consumed by FluxDevArchAdapter / SDXLArchAdapter (future PRs).
        """
        ...

    def default_steps(self) -> int:
        """Default num_inference_steps when caller didn't specify.
        Klein default = 25 (the Pipeline default at __call__ is 50, but distilled
        models converge in 9-25 — we pick 25 for safety).
        """
        ...

    def default_guidance_scale(self) -> float:
        """Default guidance_scale parameter for the Pipeline call.
        Distilled models ignore this but pipelines still expect the kwarg.
        Consumed by FluxDevArchAdapter (future PR). For distilled archs this
        value is functionally ignored at inference."""
        ...

    def supported_samplers(self) -> set[str]:
        """采样器(KSampler `sampler_name` 下拉)中本架构**真能用**的子集。

        通用 KSampler 节点列出候选(euler/heun/lcm…,可扩展);运行时 ModularImageBackend
        据此校验 —— 选了不在此集合的 → fail loud(清晰报错,不出图)。真模型已验:diffusers
        Flux2 flow-matching 管线只有 euler 兼容(heun 的 set_timesteps 不接 custom sigmas;
        lcm 只对 LCM-蒸馏模型有效)。新架构接入时在其 adapter 里声明支持集。"""
        ...

    def supported_schedulers(self) -> set[str]:
        """调度器(KSampler `scheduler` 下拉,sigma 调度)中本架构真能用的子集。
        Flux2:normal/karras/exponential/beta 均经 FlowMatchEuler 真模型验过。"""
        ...


class FluxKleinArchAdapter:
    """Flux2-Klein-9B family via diffusers Flux2KleinPipeline (标准 pipeline)。

    注:supports_cfg / supports_negative_prompt 是 legacy 字段(当前未被引擎消费)。#144 起
    comfy 单文件走标准 Flux2KleinPipeline(is_distilled=False)→ cfg/negative **实际生效**
    (true-CFG),与下面的 False 不符;留待架构收口 spec 统一清理,先不动以免破坏既有测试。"""

    def supports_cfg(self) -> bool:
        return False

    def supports_negative_prompt(self) -> bool:
        return False

    def default_steps(self) -> int:
        return 25

    def default_guidance_scale(self) -> float:
        return 4.0  # matches Pipeline kwarg default; ignored at inference for distilled

    def supported_samplers(self) -> set[str]:
        # diffusers Flux2 flow-matching 管线只有 euler 兼容(真模型验:heun set_timesteps
        # 不接 custom sigmas → 崩;lcm 仅 LCM-蒸馏模型有效)。
        return {"euler"}

    def supported_schedulers(self) -> set[str]:
        # 9 个全支持(对齐 ComfyUI):normal/karras/exponential/beta 走 diffusers use_*_sigmas;
        # simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal 手动算 sigma 注入
        # pipe(sigmas=...)。sigma 算法 port 自 ComfyUI,数值经 ground truth 验过
        # (sigma_schedules.py + test_sigma_schedules.py)。
        return {
            "normal", "karras", "exponential", "beta",
            "simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal",
        }


# Registry — key is the diffusers Pipeline class name as returned by
# Pipeline.__class__.__name__ (or read from model_index.json _class_name).
# PR-2 only registers FluxKlein; future PRs add more entries.
MODEL_ARCH_REGISTRY: dict[str, ModelArchAdapter] = {
    "Flux2KleinPipeline": FluxKleinArchAdapter(),
}
