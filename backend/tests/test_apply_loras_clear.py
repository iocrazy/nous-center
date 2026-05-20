"""PR-4 final-review I1: _apply_loras empty list must clear a foreign LoRA
active on a shared base transformer, and must NOT crash on a fresh pipe."""
from __future__ import annotations

from src.services.inference.image_diffusers import DiffusersImageBackend


def _bare_adapter(pipe):
    a = DiffusersImageBackend.__new__(DiffusersImageBackend)
    a._pipe = pipe
    a._loaded_loras = set()
    a._offload_strategy = "no_offload"
    return a


def test_apply_loras_empty_clears_foreign_active_adapter():
    """No-LoRA combo must deactivate a LoRA another adapter left active on the
    shared base transformer (PR-4 final-review I1)."""
    class _Pipe:
        _is_offloaded = False
        def __init__(self): self.active = ["foreign"]
        def get_active_adapters(self): return list(self.active)
        def set_adapters(self, names, adapter_weights=None): self.active = list(names)
    a = _bare_adapter(_Pipe())
    a._apply_loras([])
    assert a._pipe.active == []


def test_apply_loras_empty_noop_on_fresh_pipe():
    """Fresh pipe with zero registered adapters: must NOT call set_adapters([])
    (diffusers raises KeyError there)."""
    class _Pipe:
        _is_offloaded = False
        def get_active_adapters(self): return []
        def set_adapters(self, *a, **k): raise AssertionError("set_adapters must not be called on a fresh pipe")
    a = _bare_adapter(_Pipe())
    a._apply_loras([])  # must not raise
