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
from typing import Any, Callable

import torch
from safetensors.torch import load_file

from src.services.inference.component_spec import ComponentSpec

logger = logging.getLogger(__name__)


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

    def dispatch(self, spec: ComponentSpec) -> Any:
        for matcher, fn in self._loaders:
            if matcher(spec):
                logger.debug("quant_loaders: dispatching %s to %s", spec.file, fn.__name__)
                return fn(spec)
        raise UnsupportedQuantError(f"no quant loader matches {spec.file!r}")


QUANT_LOADERS = QuantLoaderRegistry()


# Reject GGUF eagerly — V2 PR-7 work, not in scope for PR-1.
@QUANT_LOADERS.register(match=lambda spec: spec.file.lower().endswith(".gguf"))
def reject_gguf(spec: ComponentSpec) -> Any:
    raise UnsupportedQuantError(
        f"GGUF quantization is V2 PR-7 follow-up; cannot load {spec.file!r} in PR-1"
    )


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,  # torch 2.10 native
    }.get(dtype_str, torch.bfloat16)


# Plain bf16/fp16 safetensors — uniform state_dict loader. Caller (PR-2's
# DiffusersImageBackend or test) decides whether to wrap into a module.
@QUANT_LOADERS.register(match=lambda spec: spec.file.endswith(".safetensors"))
def load_safetensors_plain(spec: ComponentSpec) -> dict[str, torch.Tensor]:
    """Plain bf16/fp16 safetensors → state_dict, target dtype applied.

    Note: this is the FALLBACK matcher in the registry. Specific formats (fp8mixed,
    mxfp8mixed, nvfp4mixed) registered LATER in this module will match first via
    filename substring; this loader only runs for safetensors without those markers.
    """
    target = _dtype_str_to_torch(spec.dtype)
    sd = load_file(spec.file, device="cpu")  # always load to CPU first; .to(device) is caller's job
    return {k: v.to(target) for k, v in sd.items()}
