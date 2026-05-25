"""get_or_load_image_adapter(PR-4 起 modular-only)—— auto 解析 + combo 缓存 + 四态事件。

PR-4 删了 legacy 组件 L1 缓存(get_or_load_component / DiffusersImageBackend);本文件改测
modular 路径(_get_or_load_modular_adapter),mock 掉真 build seam(ModularImageBackend /
_import_modular / repo 推导)。实际出图由真模型 smoke 验。
"""
from __future__ import annotations

import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


def _comps(unet_dev="cuda:1"):
    return {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device=unet_dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.fixture
def stubbed(monkeypatch):
    """mock modular build seam:repo 推导 + ModularImageBackend(无真 pipe)。"""
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)  # 跳过 VRAM 守卫
    monkeypatch.setattr(MM, "_modular_repo_from_components", lambda resolved: "/fake/repo")
    monkeypatch.setattr(MM, "_is_comfy_single_file_unet", lambda spec: False)
    monkeypatch.setattr(IM, "_import_modular", lambda: (object(), lambda: object()))

    builds = []
    fail = {"on": False}

    class _FakeBackend:
        def __init__(self, **kw):
            builds.append(kw)

        async def load(self, dev):
            pass

        def _ensure_pipe(self):
            if fail["on"]:
                raise RuntimeError("synthetic build fail")

    monkeypatch.setattr(IM, "ModularImageBackend", _FakeBackend)
    return mm, builds, fail


@pytest.mark.asyncio
async def test_same_combo_cache_hit(stubbed):
    mm, builds, _ = stubbed
    a1 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    a2 = await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline")
    assert a1 is a2
    assert len(builds) == 1  # 第二次走 combo 缓存,不重建


@pytest.mark.asyncio
async def test_auto_device_resolved(stubbed, monkeypatch):
    mm, builds, _ = stubbed
    monkeypatch.setattr(mm._allocator, "get_best_gpu", lambda vram: 2)
    await mm.get_or_load_image_adapter(_comps(unet_dev="auto"), "Flux2KleinPipeline")
    assert builds[0]["device"] == "cuda:2"  # auto → cuda:2


@pytest.mark.asyncio
async def test_emits_component_events(stubbed):
    from src.services.inference.component_spec import component_state_key
    mm, _builds, _ = stubbed
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    comps = _comps()
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline", on_event=_on_event)
    unet_dev = mm._resolve_component_device(comps["diffusion_models"]).device
    keys = {component_state_key(comps[k].model_copy(update={"device": unet_dev})) for k in ("diffusion_models", "clip", "vae")}
    loaded = {k for (k, s, _e) in events if s == "loaded"}
    loading = {k for (k, s, _e) in events if s == "loading"}
    assert keys <= loaded and keys <= loading


@pytest.mark.asyncio
async def test_emits_failed_on_build_error(stubbed):
    mm, _builds, fail = stubbed
    fail["on"] = True
    events = []

    async def _on_event(key, state, error):
        events.append((key, state, error))

    with pytest.raises(RuntimeError):
        await mm.get_or_load_image_adapter(_comps(), "Flux2KleinPipeline", on_event=_on_event)
    assert any(s == "failed" for (_k, s, _e) in events)
