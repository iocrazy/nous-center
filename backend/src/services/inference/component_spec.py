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
_OFFLOAD_RE = re.compile(r"^(none|cpu|stream|cuda:(0|[1-9]\d*))$")  # stream=流式分块(lowvram spec 2026-06-12)


class ComponentSpec(BaseModel):
    """One Flux2 / SDXL / Z-Image component (transformer | text encoder | vae).

    `device` accepts "auto" → ModelManager.get_best_gpu(vram_mb) resolves at load time.
    `loras` is meaningful only when kind="diffusion_models" (Flux2 LoRAs patch DiT, not text_encoder/VAE).
    """

    kind: Literal["diffusion_models", "clip", "vae"]
    file: str = Field(..., description="Absolute path resolved by component_scanner")
    device: str = Field(..., description="'auto' | 'cpu' | 'cuda:N'")
    dtype: str = Field(..., description="'bfloat16' | 'float16' | 'fp8_e4m3'")
    # 逐组件放置/卸载策略(逐组件跨卡 spec 2026-06-04):none=常驻 device;cpu=不用时挪 CPU
    # (enable_model_cpu_offload 等价,塞大模型);cuda:N=跨卡 stash。**不进** to_component_key /
    # component_state_key —— offload 是放置策略,不改变「加载了哪个权重」的身份。
    offload: str = Field("none", description="'none' | 'cpu' | 'cuda:N'")
    loras: list[LoRASpec] = Field(default_factory=list)
    adapter_arch: str | None = Field(None, description="unet only: 'flux2' | 'flux1'")
    # Ideogram-4 双 DiT(非对称 CFG):diffusion_models 单文件携带第二个(unconditional)DiT 单文件路径。
    # runner 据此建 unconditional_transformer override(spec 2026-06-12)。其余架构 None(零回归)。
    unconditional_file: str | None = Field(None, description="ideogram4 only: second (unconditional) DiT single-file")
    clip_arch: str | None = Field(None, description="clip only: 'flux2' | 'flux1' | 'sdxl' | 'qwen'")

    @field_validator("offload")
    @classmethod
    def _validate_offload(cls, v: str) -> str:
        v = v or "none"
        if not _OFFLOAD_RE.match(v):
            raise ValueError(f"offload must match none|cpu|stream|cuda:N (no leading zero) — got {v!r}")
        if v.startswith("cuda:"):
            idx = int(v.split(":", 1)[1])
            return f"cuda:{idx}"
        return v

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
        if self.loras and self.kind != "diffusion_models":
            raise ValueError(
                f"loras is only meaningful for kind='diffusion_models', got kind={self.kind!r}"
            )
        if self.adapter_arch is not None and self.kind != "diffusion_models":
            raise ValueError(
                f"adapter_arch is diffusion_models-only, got kind={self.kind!r}"
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
ComponentKey = tuple[str, str, str, frozenset[tuple[str, float]], str | None]


def to_component_key(spec: ComponentSpec) -> ComponentKey:
    """Compute the L1 cache key for this component.

    LoRA list → frozenset of (name, strength) so re-ordering doesn't break cache hits.
    Two specs with identical file/device/dtype + same LoRAs (any order) produce equal keys.
    """
    lora_set = frozenset((lora.name, float(lora.strength)) for lora in spec.loras)
    # 第 5 元 = unconditional_file(ideogram4 双 DiT;None 不影响其余架构身份,零回归)。
    # 不同 uncond DiT 配同 cond → 不同 combo,避免错命中。
    return (spec.file, spec.device, spec.dtype, lora_set, spec.unconditional_file)


def component_state_key(spec: ComponentSpec) -> str:
    """Stable wire/UI string key for one component's load state. Derived from
    to_component_key so it matches the L1 cache identity: file|device|dtype|loras.
    LoRAs are sorted (order-independent) as 'name@strength' joined by '+'. The
    frontend (PR-5b) computes the identical string from the loader-node descriptor."""
    file, device, dtype, lora_set, _uncond = to_component_key(spec)
    lora_sig = "+".join(sorted(f"{name}@{strength}" for name, strength in lora_set))
    # uncond DiT 不进 state_key 串(cond file 已唯一标识双 DiT 对;保持前端四态匹配串不变,零回归)。
    return f"{file}|{device}|{dtype}|{lora_sig}"


# Resolve ImageRequest's forward ref to ComponentSpec now that both classes
# exist (base.py only TYPE_CHECKING-imports this module to avoid a cycle).
from src.services.inference.base import ImageRequest as _ImageRequest  # noqa: E402
_ImageRequest.model_rebuild()
