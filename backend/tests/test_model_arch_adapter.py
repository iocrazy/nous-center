"""ModelArchAdapter Protocol conformance + FluxKlein dispatch."""
from __future__ import annotations

from src.services.inference.model_arch_adapter import (
    ModelArchAdapter,
    MODEL_ARCH_REGISTRY,
    FluxKleinArchAdapter,
)


def test_flux_klein_adapter_in_registry():
    """FluxKlein adapter must be registered under the diffusers Pipeline class name."""
    assert "Flux2KleinPipeline" in MODEL_ARCH_REGISTRY
    adapter = MODEL_ARCH_REGISTRY["Flux2KleinPipeline"]
    assert isinstance(adapter, FluxKleinArchAdapter)


def test_flux_klein_adapter_supports_cfg_false():
    """Klein is distilled — no CFG branch."""
    adapter = FluxKleinArchAdapter()
    assert adapter.supports_cfg() is False


def test_flux_klein_adapter_supports_negative_prompt_false():
    """Klein doesn't accept negative_prompt (distilled inference is positive-prompt-only)."""
    adapter = FluxKleinArchAdapter()
    assert adapter.supports_negative_prompt() is False


def test_flux_klein_adapter_default_steps():
    """Klein default steps = 25 (distilled but supports 9-50 in practice)."""
    adapter = FluxKleinArchAdapter()
    assert adapter.default_steps() == 25


def test_flux_klein_adapter_default_guidance_scale():
    """Klein guidance is ignored at inference but registered for parameter pass-through."""
    adapter = FluxKleinArchAdapter()
    assert adapter.default_guidance_scale() == 4.0  # matches Pipeline default


def test_flux_klein_supported_samplers_euler_only():
    """PR-2:diffusers Flux2 flow-matching 只 euler 兼容(真模型验 heun 崩/lcm 错)。"""
    adapter = FluxKleinArchAdapter()
    assert adapter.supported_samplers() == {"euler"}


def test_flux_klein_supported_schedulers():
    """对齐 ComfyUI 9 个:normal/karras/exponential/beta(diffusers use_*_sigmas)+
    simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal(手动 sigma 注入)。
    9 个均真模型 smoke 出图验过(smoke_schedulers.py)。"""
    adapter = FluxKleinArchAdapter()
    assert adapter.supported_schedulers() == {
        "normal", "karras", "exponential", "beta",
        "simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal",
    }


def test_unknown_pipeline_class_not_in_registry():
    assert "StableDiffusionXLPipeline" not in MODEL_ARCH_REGISTRY
    assert "Flux2Pipeline" not in MODEL_ARCH_REGISTRY  # FluxDev — V2 PR


def test_protocol_can_type_check_adapter():
    """@runtime_checkable allows isinstance check at runtime (validates structural conformance)."""
    adapter = FluxKleinArchAdapter()
    assert isinstance(adapter, ModelArchAdapter)
