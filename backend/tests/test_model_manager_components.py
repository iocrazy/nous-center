"""PR-1 Task 6: ModelManager component-level cache.

Per spec §5.5, _components: dict[ComponentKey, LoadedComponent] coexists with
the legacy _models: dict[str, LoadedModel] in PR-1. PR-2 image adapters will
route through _components.

PR-4 Task 5: _load_component_impl 改返 GPU 模块包；测试改用 _load_component_module seam。
"""
from __future__ import annotations

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    """Empty stub registry — component path is registry-agnostic, so we don't
    need yaml/disk access. Bypasses ModelRegistry.__init__ which would try to
    read a config file."""

    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm():
    """Fresh ModelManager with no specs registered (component path is registry-agnostic)."""
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


@pytest.mark.asyncio
async def test_is_component_loaded_cold_by_default(mm):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    assert mm.is_component_loaded(spec) == "cold"


@pytest.mark.asyncio
async def test_get_or_load_component_marks_loaded(mm, monkeypatch):
    """_load_component_impl 经 _load_component_module seam 包装后写入缓存。"""
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    def _fake_module(s):
        return {"module": "stub", "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _fake_module)

    result = await mm.get_or_load_component(spec)
    assert result["spec"] is spec
    assert mm.is_component_loaded(spec) == "loaded"


@pytest.mark.asyncio
async def test_get_or_load_component_cache_hit_does_not_call_loader_twice(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    calls = []

    def _counting(s):
        calls.append(s)
        return {"module": f"stub{len(calls)}", "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _counting)

    r1 = await mm.get_or_load_component(spec)
    r2 = await mm.get_or_load_component(spec)
    assert r1 is r2
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_get_or_load_component_distinguishes_lora_set(mm, monkeypatch):
    """Same file+device, different LoRA list → distinct cache entries."""
    s_a = ComponentSpec(kind="unet", file="/p/u.safe", device="cuda:0", dtype="bfloat16",
                       loras=[LoRASpec(name="style", strength=0.8)])
    s_b = ComponentSpec(kind="unet", file="/p/u.safe", device="cuda:0", dtype="bfloat16",
                       loras=[LoRASpec(name="style", strength=0.4)])

    def _loader(s):
        return {"module": f"variant_{hash(frozenset((lora.name, lora.strength) for lora in s.loras))}", "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _loader)

    r_a = await mm.get_or_load_component(s_a)
    r_b = await mm.get_or_load_component(s_b)
    assert r_a["module"] != r_b["module"]


@pytest.mark.asyncio
async def test_unload_component_clears_cache_entry(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    def _loader(s): return {"module": "stub", "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _loader)

    await mm.get_or_load_component(spec)
    assert mm.is_component_loaded(spec) == "loaded"

    await mm.unload_component(spec)
    assert mm.is_component_loaded(spec) == "cold"


def test_legacy_models_dict_untouched(mm):
    """PR-1 invariant: existing _models dict, locks, load_failures all unchanged in shape."""
    assert hasattr(mm, "_models")
    assert isinstance(mm._models, dict)
    assert hasattr(mm, "_locks")
    assert hasattr(mm, "_load_failures")


@pytest.mark.asyncio
async def test_is_component_loaded_failed_when_loader_raises(mm, monkeypatch):
    spec = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")

    def _broken_loader(s):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(mm, "_load_component_module", _broken_loader)

    with pytest.raises(RuntimeError, match="synthetic OOM"):
        await mm.get_or_load_component(spec)
    assert mm.is_component_loaded(spec) == "failed"


@pytest.mark.asyncio
async def test_load_component_impl_returns_module_bundle(mm, monkeypatch):
    """_load_component_impl 经 _load_component_module seam 返回 GPU 模块包。"""
    spec = ComponentSpec(kind="vae", file="/m/v.safe", device="cuda:0", dtype="bfloat16")

    sentinel = object()

    def _fake_module(s):
        assert s is spec
        return {"module": sentinel, "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _fake_module)
    bundle = await mm.get_or_load_component(spec)
    assert bundle["module"] is sentinel
    assert bundle["spec"] is spec
    assert bundle["device"] == "cuda:0"
    assert "loaded_at" in bundle
