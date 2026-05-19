"""Quant loader registry for image-component-multi-gpu PR-1.

Per spec §5.3: registry maps (ComponentSpec) → loaded weights (state_dict).
First-match-wins; register specific formats (fp8mixed / mxfp8mixed / nvfp4mixed)
before plain safetensors fallback.

Each loader returns:
  dict[str, Tensor]      (state_dict — caller's responsibility to wrap into a module)

GGUF is rejected eagerly with UnsupportedQuantError; V2 PR-7 will add it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NoReturn

import torch
from safetensors.torch import load_file

from src.services.inference.component_spec import ComponentSpec

logger = logging.getLogger(__name__)


# Loaders all return the same shape: caller wraps into a torch.nn.Module.
StateDict = dict[str, torch.Tensor]


class UnsupportedQuantError(RuntimeError):
    """Raised when no registered loader matches a ComponentSpec."""


class QuantLoaderRegistry:
    """First-match-wins registry. Register specific formats before generic fallbacks."""

    def __init__(self) -> None:
        self._loaders: list[tuple[Callable[[ComponentSpec], bool], Callable[[ComponentSpec], Any]]] = []

    def register(self, *, match: Callable[[ComponentSpec], bool]) -> Callable[[Callable], Callable]:
        """Decorator. `match(spec)` → bool decides if this loader handles the spec."""
        def deco(fn: Callable[[ComponentSpec], Any]) -> Callable[[ComponentSpec], Any]:
            self._loaders.append((match, fn))
            return fn
        return deco

    def dispatch(self, spec: ComponentSpec) -> StateDict:
        for matcher, fn in self._loaders:
            if matcher(spec):
                logger.debug("quant_loaders: dispatching %s to %s", spec.file, fn.__name__)
                return fn(spec)
        raise UnsupportedQuantError(f"no quant loader matches {spec.file!r}")


QUANT_LOADERS = QuantLoaderRegistry()


# Reject GGUF eagerly — V2 PR-7 work, not in scope for PR-1.
@QUANT_LOADERS.register(match=lambda spec: spec.file.lower().endswith(".gguf"))
def reject_gguf(spec: ComponentSpec) -> NoReturn:
    raise UnsupportedQuantError(
        f"GGUF quantization is V2 PR-7 follow-up; cannot load {spec.file!r} in PR-1"
    )


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Map ComponentSpec.dtype string → torch.dtype.

    Raises UnsupportedQuantError on unknown dtype rather than silently falling
    back to bfloat16 (which would miscast user-loaded weights). PR-3+ adding
    a new format must register here.
    """
    try:
        return _DTYPE_MAP[dtype_str]
    except KeyError:
        raise UnsupportedQuantError(
            f"unknown target dtype {dtype_str!r}; expected one of {sorted(_DTYPE_MAP)}"
        )


def _has_comfy_quant_metadata(file_path: str) -> bool:
    """Sniff a safetensors header for any `.comfy_quant` suffixed key (cheap — no full read).

    Fails soft (returns False) on any error — caller falls through to plain loader.
    Operators can grep logs at DEBUG level to diagnose misdispatch.
    """
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".comfy_quant"):
                    return True
    except Exception as exc:  # noqa: BLE001 — fail-soft on header read error
        logger.debug("comfy_quant header sniff failed for %s: %s", file_path, exc)
        return False
    return False


@QUANT_LOADERS.register(match=lambda spec: "mxfp8mixed" in Path(spec.file).name.lower())
def load_mxfp8mixed(spec: ComponentSpec) -> StateDict:
    """Microscaling fp8 → dequant by block-wise E8M0 scale → target dtype.

    Format: per-32-element blocks, each with a uint8 E8M0 exponent in `.weight_scale`.
    Real file: Flux2-Klein-9B-True-v2-mxfp8mixed.safetensors (9.7GB).

    Algorithm:
      1. Load fp8 weight + uint8 scale tensor (1 byte per 32-element block)
      2. For each block: scale_fp32 = 2.0 ** (uint8_scale - 127)   # E8M0 with bias 127
      3. fp32_weight = fp8_weight × scale (broadcast within block)
      4. Cast to target dtype, drop metadata keys
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = load_file(spec.file, device="cpu")
    BLOCK_SIZE = 32

    clean: dict[str, torch.Tensor] = {}
    dequant_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale_uint8 = raw.get(scale_key)
            if scale_uint8 is None:
                logger.warning("mxfp8: tensor %s missing %s; using fp8 cast only", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # E8M0: scale = 2^(uint8 - 127)
            scale_fp32 = torch.pow(2.0, scale_uint8.to(torch.float32) - 127.0)
            # Broadcast block-wise: flatten weights, repeat each scale BLOCK_SIZE times
            flat = tensor.flatten().to(torch.float32)
            assert flat.numel() % BLOCK_SIZE == 0, \
                f"mxfp8: {key} numel {flat.numel()} not divisible by block {BLOCK_SIZE}"
            assert scale_fp32.numel() * BLOCK_SIZE == flat.numel(), \
                f"mxfp8: {key} scale count {scale_fp32.numel()} × block {BLOCK_SIZE} ≠ weight numel {flat.numel()}"
            block_scales = scale_fp32.repeat_interleave(BLOCK_SIZE)
            dequant = (flat * block_scales).to(target).reshape(tensor.shape)
            clean[key] = dequant
            dequant_count += 1
        else:
            # Same pass-through policy as load_fp8mixed: only cast fp32/fp16
            if tensor.dtype in (torch.float32, torch.float16):
                clean[key] = tensor.to(target)
            else:
                clean[key] = tensor

    logger.info("quant_loaders.mxfp8mixed: %d block-quant tensors dequant'd, %d total keys (%s)",
                dequant_count, len(clean), Path(spec.file).name)
    return clean


@QUANT_LOADERS.register(match=lambda spec: (
    "fp8mixed" in Path(spec.file).name.lower()
    or _has_comfy_quant_metadata(spec.file)
))
def load_fp8mixed(spec: ComponentSpec) -> StateDict:
    """Wikeeyang comfy_quant fp8 → dequant by `.weight_scale` companion → target dtype.

    Algorithm (preserved from image_diffusers.py:105 load_quantized_transformer):
      1. safetensors_load_file → state dict with fp8 weights + .weight_scale + .comfy_quant
      2. For each fp8 tensor, multiply by float32 scale, cast to target dtype
      3. Drop .weight_scale and .comfy_quant marker keys
      4. Return clean state dict ready for caller's load_state_dict

    Reference fixture: /media/heygo/Program/models/nous/image/diffusion_models/
    Flux2-Klein-9B-True-v2-fp8mixed.safetensors
    """
    target = _dtype_str_to_torch(spec.dtype)
    raw = load_file(spec.file, device="cpu")

    clean: dict[str, torch.Tensor] = {}
    fp8_count = 0
    for key, tensor in raw.items():
        if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
            continue  # metadata key, drop
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale"
            scale = raw.get(scale_key)
            if scale is None:
                logger.warning("fp8 tensor %s has no companion %s scale; loading at fp8 dtype", key, scale_key)
                clean[key] = tensor.to(target)
                continue
            # dequant: fp8 × scale → fp32 → target
            clean[key] = (tensor.to(torch.float32) * scale.to(torch.float32)).to(target)
            fp8_count += 1
        else:
            # Preserve pre-refactor behavior: only cast fp32/fp16 tensors to target;
            # other dtypes (uint8/int/bool marker tensors) pass through unchanged.
            # On Flux2 fp8mixed in practice the only non-fp8 tensors are bf16 weights,
            # so this is a no-op vs the unconditional cast — but it makes the
            # "byte-identical to pre-PR-1" claim in image_diffusers.py:129 honest.
            if tensor.dtype in (torch.float32, torch.float16):
                clean[key] = tensor.to(target)
            else:
                clean[key] = tensor

    logger.info("quant_loaders.fp8mixed: %d fp8 weights dequant'd, %d total keys (%s)",
                fp8_count, len(clean), Path(spec.file).name)
    return clean


# Plain bf16/fp16 safetensors — uniform state_dict loader. Caller (PR-2's
# DiffusersImageBackend or test) decides whether to wrap into a module.
@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec) -> StateDict:
    """Plain bf16/fp16 safetensors → state_dict, target dtype applied.

    Note: this is the FALLBACK matcher. It MUST stay last in this module —
    PR-3+ fp8mixed / mxfp8mixed / nvfp4mixed loaders register ABOVE this function
    so their filename-substring matchers run first (`_loaders` is iterated in
    registration order; first match wins). Loads to CPU regardless of
    `spec.device` — caller is responsible for the subsequent `.to(device)`.
    """
    target = _dtype_str_to_torch(spec.dtype)
    sd = load_file(spec.file, device="cpu")
    return {k: v.to(target) for k, v in sd.items()}
