"""组件 L1 PR-2c:SeedVR2(by-key 模型)常驻 pin —— 协议 + runner handler + client + 端点 + mm 行为。

SeedVR2 不在 registry(by-key,id=image:SeedVR2:<hash>),老 /resident 改 yaml 够不着。本 PR 加
跨进程 SetModelResident:切 runner _models 里 by-key 模型的常驻位(in-memory,resident=True 不被 LRU 驱逐)。
CI 安全:protocol/client/engines/runner 源码检查;mm.set_model_resident 真行为用合成 LoadedModel。
"""
from __future__ import annotations

import pathlib

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager

_SRC = pathlib.Path(__file__).parent.parent / "src"


def test_protocol_message():
    from src.runner import protocol as P  # noqa: PLC0415

    m = P.SetModelResident(model_id="image:SeedVR2:abc", resident=True)
    assert m.kind == "set_model_resident"
    assert P._KIND_TO_CLASS.get("set_model_resident") is P.SetModelResident
    assert P.SetModelResident in P.Message.__args__
    back = P.decode(P.encode(m))
    assert isinstance(back, P.SetModelResident) and back.resident is True


def test_runner_handler_wired():
    src = (_SRC / "runner/runner_process.py").read_text()
    assert "_handle_set_model_resident(" in src
    assert "set_model_resident(" in src
    assert "isinstance(msg, P.SetModelResident)" in src


def test_runner_client_has_method():
    src = (_SRC / "runner/client.py").read_text()
    assert "async def set_model_resident(" in src
    assert "P.SetModelResident(" in src


def test_engines_route_registered():
    src = (_SRC / "api/routes/engines.py").read_text()
    assert '"/seedvr2/resident"' in src
    assert "sup.client.set_model_resident(" in src


# ---- mm.set_model_resident 真行为(合成 LoadedModel,frozen spec 经 model_copy) -------

class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


from src.services.inference.base import InferenceAdapter as _IA


class _FakeAdapter(_IA):
    def __init__(self):
        self._model = object()  # is_loaded → True

    async def load(self, dev):  # pragma: no cover
        pass

    async def infer(self, req):  # pragma: no cover
        raise NotImplementedError

    def unload(self):
        self._model = None


def _put_seedvr2(mm, model_id="image:SeedVR2:abc"):
    from src.services.model_manager import LoadedModel  # noqa: PLC0415
    spec = mm._synthesize_image_spec(
        model_id=model_id,
        adapter_class_path="src.services.inference.image_seedvr2.SeedVR2UpscaleBackend",
        target_device="cuda:1", vram_mb=8000, pipeline_class="SeedVR2", source_files=["dit.safetensors"])
    mm._models[model_id] = LoadedModel(spec=spec, adapter=_FakeAdapter(), gpu_index=1, gpu_indices=[1])
    return model_id


def test_set_model_resident_flips_frozen_spec():
    """frozen ModelSpec → model_copy 换新 spec;resident 翻 True/False。"""
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    mid = _put_seedvr2(mm)
    assert mm._models[mid].spec.resident is False  # 合成默认非常驻

    assert mm.set_model_resident(mid, True) is True
    assert mm._models[mid].spec.resident is True
    assert mm.set_model_resident(mid, False) is True
    assert mm._models[mid].spec.resident is False


def test_set_model_resident_unknown_is_noop():
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    assert mm.set_model_resident("image:SeedVR2:nope", True) is False


async def _evict_after_pin():
    """resident=True 的 SeedVR2 不被 evict_lru 驱逐(spec.resident 被守卫读取)。"""
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    mid = _put_seedvr2(mm)
    mm.set_model_resident(mid, True)
    evicted = await mm.evict_lru(gpu_index=1)
    return evicted, mid


def test_resident_seedvr2_not_evicted():
    import asyncio  # noqa: PLC0415
    evicted, mid = asyncio.run(_evict_after_pin())
    assert evicted is None  # resident → evict_lru 跳过(候选里被 spec.resident 过滤)
