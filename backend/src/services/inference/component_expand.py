"""PR-4 §7.4: legacy image model_key → 3 ComponentSpec (backend inline expand).

Old workflows reference image_generate by model_key only. The runner component
path needs unet/clip/vae descriptors, so before dispatch we translate the
model's yaml ModelSpec into 3 ComponentSpec (device='auto') and fold any
legacy LoRA list into the unet descriptor. File paths follow the HF layout
<root>/{transformer,text_encoder,vae}/ where <root> = paths['main'];
quantized_transformer overrides the unet file.
"""
from __future__ import annotations

import glob
from pathlib import Path

from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec


def _abs(rel_or_abs: str) -> Path:
    from src.config import get_settings
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return Path(get_settings().LOCAL_MODELS_PATH) / p


def _representative_file(component_dir: Path) -> str:
    """Pick a real .safetensors inside the HF component dir so Path(file).parent
    == the dir (load_component_module does from_pretrained(parent)). Falls back
    to a synthetic path under the dir if none on disk (load will then error
    clearly rather than silently mis-pathing)."""
    hits = sorted(glob.glob(str(component_dir / "*.safetensors")))
    if hits:
        return hits[0]
    return str(component_dir / "model.safetensors")


def expand_legacy_image_spec(spec, loras: list[dict] | None = None) -> dict[str, ComponentSpec]:
    main = _abs(spec.paths["main"])
    arch = (spec.params.get("accepts_lora_archs") or ["flux2"])[0]

    qt = spec.paths.get("quantized_transformer")
    unet_file = str(_abs(qt)) if qt else _representative_file(main / "transformer")
    clip_file = _representative_file(main / "text_encoder")
    vae_file = _representative_file(main / "vae")

    lora_specs = [
        LoRASpec(name=lo["name"], strength=float(lo.get("strength", 1.0)), path=lo.get("path"))
        for lo in (loras or []) if isinstance(lo, dict) and lo.get("name")
    ]
    return {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=unet_file, device="auto", dtype="bfloat16",
                              adapter_arch=arch, loras=lora_specs),
        "clip": ComponentSpec(kind="clip", file=clip_file, device="auto", dtype="bfloat16", clip_arch=arch),
        "vae":  ComponentSpec(kind="vae",  file=vae_file,  device="auto", dtype="bfloat16"),
    }
