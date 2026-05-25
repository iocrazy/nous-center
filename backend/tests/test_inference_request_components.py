"""PR-4: ImageRequest.components + LoRASpec.path round-trip."""
from __future__ import annotations

from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference.component_spec import ComponentSpec


def test_lora_spec_path_optional_default_none():
    assert LoRASpec(name="style", strength=0.8).path is None
    s = LoRASpec(name="style", strength=0.8, path="/m/loras/style.safetensors")
    assert s.path == "/m/loras/style.safetensors"


def test_image_request_components_default_none():
    req = ImageRequest(request_id="r1", prompt="a cat")
    assert req.components is None
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_image_request_with_components():
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }
    req = ImageRequest(request_id="r1", prompt="a cat", seed=42, components=comps)
    assert req.components["diffusion_models"].device == "cuda:1"
    assert req.components["vae"].kind == "vae"
