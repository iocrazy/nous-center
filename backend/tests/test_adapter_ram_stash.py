"""adapter 级 RAM stash(spec 2026-06-12 PR-2):整模型路线 stash/restore + 守卫 + 记账。

bookkeeping/守卫单测(fake adapter/pipe);真权重搬运与时延由真机验
(组件层 PR-1 真机:restore 2.3s/组件;fp8 .to 往返 bit 一致 spike 已验)。
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

import src.services.inference.image_modular as IM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import InferenceAdapter
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import LoadedModel, ModelManager


class _Reg(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


class _FakeAdapter(InferenceAdapter):
    modality = None
    estimated_vram_mb = 0

    def __init__(self, stash_ok=True):
        self._stash_ok = stash_ok
        self.stash_calls = 0
        self.restore_calls = 0
        self._model = object()

    async def load(self, device):  # pragma: no cover
        pass

    async def infer(self, req):  # pragma: no cover
        raise NotImplementedError

    def stash(self):
        self.stash_calls += 1
        return self._stash_ok

    def restore(self):
        self.restore_calls += 1


def _spec(mid="image:ZImagePipeline:z:abc", vram=1000, resident=False):
    return ModelSpec(id=mid, model_type="image", adapter_class="modular",
                     paths={"main": "/m/x"}, vram_mb=vram, resident=resident)


def _entry(mm, adapter=None, mid="image:ZImagePipeline:z:abc", **kw):
    a = adapter or _FakeAdapter()
    e = LoadedModel(spec=_spec(mid, **kw), adapter=a, gpu_index=1)
    mm._models[mid] = e
    return e, a


@pytest.fixture
def mm(monkeypatch):
    m = ModelManager(registry=_Reg(), allocator=GPUAllocator())
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=100 * 10**9))
    return m


@pytest.mark.asyncio
async def test_stash_model_marks_and_calls_adapter(mm):
    e, a = _entry(mm)
    assert await mm.stash_model(e.spec.id) is True
    assert e.stashed is True and a.stash_calls == 1


@pytest.mark.asyncio
async def test_stash_model_guards(mm):
    """in_use / resident / 被引用 / 已 stashed(二次=真销毁)/ 引擎不支持 → False。"""
    e, a = _entry(mm, mid="m1")
    mm._in_use.add("m1")
    assert await mm.stash_model("m1") is False
    mm._in_use.discard("m1")

    e2, _ = _entry(mm, mid="m2", resident=True)
    assert await mm.stash_model("m2") is False

    e3, _ = _entry(mm, mid="m3")
    mm._references["m3"] = {"someone"}
    assert await mm.stash_model("m3") is False

    e4, _ = _entry(mm, mid="m4")
    assert await mm.stash_model("m4") is True
    assert await mm.stash_model("m4") is False, "已 stashed 再卸 = 真销毁路径"

    e5, _ = _entry(mm, adapter=_FakeAdapter(stash_ok=False), mid="m5")
    assert await mm.stash_model("m5") is False


@pytest.mark.asyncio
async def test_stash_model_refuses_l1_combo(mm):
    """组件路线 combo(L1 池里有 refs)不在 adapter 层 stash(组件层 PR-1 已覆盖)。"""
    e, a = _entry(mm, mid="combo1")
    mm._components[("f", "cuda:1", "bf16", frozenset())] = {
        "module": object(), "role": "vae", "key": ("f", "cuda:1", "bf16", frozenset()),
        "refs": {"combo1"}, "resident": False, "last_used": time.monotonic(), "device": "cuda:1",
    }
    assert await mm.stash_model("combo1") is False
    assert a.stash_calls == 0


@pytest.mark.asyncio
async def test_stash_model_low_ram_refuses(mm, monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=1 * 10**9))
    e, a = _entry(mm)
    assert await mm.stash_model(e.spec.id) is False


@pytest.mark.asyncio
async def test_hit_restores_stashed_adapter(mm):
    """get_or_load 命中 stashed entry → adapter.restore + stashed=False。"""
    combo_key = ("Flux2KleinPipeline", ("a", "cuda:1", "bf16"), ("b", "cuda:1", "bf16"),
                 ("c", "cuda:1", "bf16"), "none", ("none", "none", "none"))
    mid = mm._derive_image_model_id(combo_key)
    e, a = _entry(mm, mid=mid)
    e.stashed = True

    from src.services.inference.component_spec import ComponentSpec
    resolved = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/a", device="cuda:1", dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", file="/m/b", device="cuda:1", dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file="/m/c", device="cuda:1", dtype="bfloat16"),
    }
    out = await mm._get_or_load_modular_adapter(resolved, combo_key, "Flux2KleinPipeline", "cuda:1",
                                                _async_noop)
    assert out is a
    assert a.restore_calls == 1 and e.stashed is False


async def _async_noop(*a, **k):
    return None


def test_evictable_excludes_stashed(mm):
    e1, _ = _entry(mm, mid="m1", vram=5000)
    e2, _ = _entry(mm, mid="m2", vram=7000)
    e2.stashed = True
    assert mm._evictable_mb_on_card(1) == 5000


def test_snapshot_reports_adapter_stashed(mm):
    e, _ = _entry(mm)
    e.stashed = True
    snap = mm.loaded_models_snapshot()
    assert snap and snap[0]["stashed"] is True


def test_engine_stash_guards(monkeypatch):
    """ModularImageBackend.stash:干净 pipe → to('cpu') True;offload/override/无 pipe → False。"""
    be = IM.ModularImageBackend(repo="/m/z", device="cuda:1", pipeline_class="ZImagePipeline")
    assert be.stash() is False, "无 pipe 不可 stash"

    pipe = MagicMock(name="pipe")
    be._pipe = pipe
    assert be.stash() is True
    pipe.to.assert_called_once_with("cpu")
    be.restore()
    pipe.to.assert_called_with("cuda:1")

    be2 = IM.ModularImageBackend(repo="/m/z", device="cuda:1", pipeline_class="ZImagePipeline",
                                 offload="cpu")
    be2._pipe = MagicMock()
    assert be2.stash() is False, "offload pipe(hook)不可整体 .to"

    be3 = IM.ModularImageBackend(repo="/m/z", device="cuda:1", pipeline_class="ZImagePipeline")
    be3._pipe = MagicMock()
    be3._transformer_override = object()
    assert be3.stash() is False, "override 装配走组件层 stash"
