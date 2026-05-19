"""ComponentSpec — single Flux2 component descriptor (unet / clip / vae).

PR-1 of image-component-multi-gpu spec §5.1. Emitted by future loader workflow nodes
(`image_unet_load` etc., added in PR-4). Cached by ModelManager via ComponentKey.

Cross-process safety: pure pydantic v2 model, msgpack-serializable through P.RunNode.inputs.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.services.inference.base import LoRASpec

_DEVICE_RE = re.compile(r"^(cpu|auto|cuda:\d+)$")


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
            raise ValueError(f"device must match cpu|auto|cuda:N — got {v!r}")
        return v


# (file, device, lora_set) — order-independent on loras via frozenset
ComponentKey = tuple[str, str, frozenset[tuple[str, float]]]


def to_component_key(spec: ComponentSpec) -> ComponentKey:
    """Compute the L1 cache key for this component.

    LoRA list → frozenset of (name, strength) so re-ordering doesn't break cache hits.
    Two specs with identical file/device + same LoRAs (any order) produce equal keys.
    """
    lora_set = frozenset((lora.name, float(lora.strength)) for lora in spec.loras)
    return (spec.file, spec.device, lora_set)
