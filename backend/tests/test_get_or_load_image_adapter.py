"""PR-4: get_or_load_image_adapter — auto 解析 + 组件级 L1 + combo 缓存。"""
from __future__ import annotations

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm():
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _comps(unet_loras=None, unet_dev="cuda:1"):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device=unet_dev, dtype="bfloat16",
                              adapter_arch="flux2", loras=unet_loras or []),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.fixture
def stubbed(mm, monkeypatch):
    module_loads = []

    def _load_module(spec):
        module_loads.append((spec.kind, spec.file, spec.device, tuple((lo.name, lo.strength) for lo in spec.loras)))
        return {"module": object(), "tokenizer": None}

    monkeypatch.setattr(mm, "_load_component_module", _load_module)

    assemble_calls = []

    def _fake_from_loaded(modules, components, pipeline_class):
        assemble_calls.append(components)
        return object()

    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(_fake_from_loaded))
    return mm, module_loads, assemble_calls


@pytest.mark.asyncio
async def test_same_combo_cache_hit(stubbed):
    mm, module_loads, assemble_calls = stubbed
    a1 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    a2 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    assert a1 is a2
    assert len(assemble_calls) == 1
    assert len(module_loads) == 3


@pytest.mark.asyncio
async def test_lora_change_reuses_all_base_modules(stubbed):
    mm, module_loads, assemble_calls = stubbed
    await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    await mm.get_or_load_image_adapter(
        _comps(unet_loras=[LoRASpec(name="s", path="/m/loras/s.safe", strength=0.8)]),
        "Flux2KleinPipeline")
    kinds = [m[0] for m in module_loads]
    assert kinds.count("clip") == 1
    assert kinds.count("vae") == 1
    assert kinds.count("unet") == 1
    assert len(assemble_calls) == 2


@pytest.mark.asyncio
async def test_auto_device_resolved(mm, monkeypatch):
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda vram: 2)
    seen = []
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: (seen.append(spec.device), {"module": object(), "tokenizer": None})[1])
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: object()))
    comps = _comps(unet_dev="auto")
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    assert "auto" not in seen
    assert "cuda:2" in seen


@pytest.mark.asyncio
async def test_image_adapter_emits_component_events(stubbed):
    from src.services.inference.component_spec import component_state_key
    mm, module_loads, assemble_calls = stubbed
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    comps = _comps()
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline", on_event=_on_event)

    # 整模型单卡:clip/vae 被统一到 unet 的卡 → 事件 key 用 unet 的 device。
    unet_dev = mm._resolve_component_device(comps["unet"]).device
    keys = {
        component_state_key(comps[k].model_copy(update={"device": unet_dev}))
        for k in ("unet", "clip", "vae")
    }
    loaded = {k for (k, s, _e) in events if s == "loaded"}
    loading = {k for (k, s, _e) in events if s == "loading"}
    assert keys <= loaded
    assert keys <= loading


@pytest.mark.asyncio
async def test_image_adapter_emits_failed_on_load_error(mm, monkeypatch):
    def _boom(spec):
        raise RuntimeError("synthetic load fail")
    monkeypatch.setattr(mm, "_load_component_module", _boom)
    events = []
    async def _on_event(key, state, error):
        events.append((key, state, error))
    with pytest.raises(RuntimeError):
        await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline", on_event=_on_event)
    assert any(s == "failed" for (_k, s, _e) in events)
