"""ComponentSpec — single Flux2 component descriptor (unet / clip / vae).

PR-1 of image-component-multi-gpu spec §5.1. Emitted by future loader workflow nodes
(`image_unet_load` etc., added in PR-4). Cached by ModelManager via ComponentKey.

Cross-process safety: pure pydantic v2 model, msgpack-serializable through P.RunNode.inputs.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.services.inference.base import LoRASpec

_DEVICE_RE = re.compile(r"^(cpu|auto|cuda:(0|[1-9]\d*))$")


class ComponentSpec(BaseModel):
    """One Flux2 / SDXL / Z-Image component (transformer | text encoder | vae).

    `device` accepts "auto" → ModelManager.get_best_gpu(vram_mb) resolves at load time.
    `loras` is meaningful only when kind="unet" (Flux2 LoRAs patch DiT, not text_encoder/VAE).
    """

    kind: Literal["unet", "clip", "vae"]
    file: str = Field(..., description="Absolute path resolved by component_scanner")
    device: str = Field(..., description="'auto' | 'cpu' | 'cuda:N'")
    dtype: str = Field(..., description="'bfloat16' | 'float16' | 'fp8_e4m3'")
    loras: list[LoRASpec] = Field(default_factory=list)
    adapter_arch: str | None = Field(None, description="unet only: 'flux2' | 'flux1'")
    clip_arch: str | None = Field(None, description="clip only: 'flux2' | 'flux1' | 'sdxl' | 'qwen'")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if not _DEVICE_RE.match(v):
            raise ValueError(f"device must match cpu|auto|cuda:N (no leading zero) — got {v!r}")
        # Canonicalize the cuda:N form so 'cuda:0' is the only representation
        # (the regex above already disallows 'cuda:00' but this belt-and-suspenders
        # the cache-key invariant: any internal mutation that bypassed the regex
        # still gets normalized before reaching to_component_key).
        if v.startswith("cuda:"):
            idx = int(v.split(":", 1)[1])
            return f"cuda:{idx}"
        return v

    @model_validator(mode="after")
    def _validate_kind_field_consistency(self) -> "ComponentSpec":
        """Per-kind field constraints (spec §5.5 rev 2):
        - LoRAs only patch the unet transformer (Flux2 / SDXL convention)
        - adapter_arch describes the unet model family
        - clip_arch describes the text encoder family
        Mis-set fields are dropped silently otherwise — fail loud here instead.
        """
        if self.loras and self.kind != "unet":
            raise ValueError(
                f"loras is only meaningful for kind='unet', got kind={self.kind!r}"
            )
        if self.adapter_arch is not None and self.kind != "unet":
            raise ValueError(
                f"adapter_arch is unet-only, got kind={self.kind!r}"
            )
        if self.clip_arch is not None and self.kind != "clip":
            raise ValueError(
                f"clip_arch is clip-only, got kind={self.kind!r}"
            )
        return self


# (file, device, dtype, lora_set) — order-independent on loras via frozenset.
# dtype is included because the loaded weights differ across target dtypes:
# the same fp8mixed safetensors dequant'd to bfloat16 vs float16 produces
# different in-memory tensors (same numerical values, different storage layout
# + different SM compute path). Cache must distinguish them.
ComponentKey = tuple[str, str, str, frozenset[tuple[str, float]]]


def to_component_key(spec: ComponentSpec) -> ComponentKey:
    """Compute the L1 cache key for this component.

    LoRA list → frozenset of (name, strength) so re-ordering doesn't break cache hits.
    Two specs with identical file/device/dtype + same LoRAs (any order) produce equal keys.
    """
    lora_set = frozenset((lora.name, float(lora.strength)) for lora in spec.loras)
    return (spec.file, spec.device, spec.dtype, lora_set)
