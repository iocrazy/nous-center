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

from typing import Protocol


class ModelArchAdapter(Protocol):
    """Per-Pipeline-class settings + behavior switches consumed by ImageSampler.

    All methods are pure (no side effects) — adapter instances are singletons
    in MODEL_ARCH_REGISTRY.
    """

    def supports_cfg(self) -> bool:
        """True if this Pipeline class uses classifier-free guidance (CFG)
        in its denoise loop. Distilled models (Klein, Z-Image-Turbo) → False.
        """
        ...

    def supports_negative_prompt(self) -> bool:
        """True if encode_prompt accepts a negative_prompt argument.
        Distilled models reject it; mainline (Dev, SDXL) accept it.
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
        Distilled models ignore this but pipelines still expect the kwarg."""
        ...


class FluxKleinArchAdapter:
    """Flux2-Klein-9B (distilled). Matches diffusers Flux2KleinPipeline."""

    def supports_cfg(self) -> bool:
        return False

    def supports_negative_prompt(self) -> bool:
        return False

    def default_steps(self) -> int:
        return 25

    def default_guidance_scale(self) -> float:
        return 4.0  # matches Pipeline kwarg default; ignored at inference for distilled


# Registry — key is the diffusers Pipeline class name as returned by
# Pipeline.__class__.__name__ (or read from model_index.json _class_name).
# PR-2 only registers FluxKlein; future PRs add more entries.
MODEL_ARCH_REGISTRY: dict[str, ModelArchAdapter] = {
    "Flux2KleinPipeline": FluxKleinArchAdapter(),
}
