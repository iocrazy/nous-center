import pytest
from unittest.mock import MagicMock
from src.services.inference.base import InferenceAdapter, InferenceResult
from src.services.inference.registry import ModelSpec
from src.services.model_manager import ModelManager


class FakeAdapter(InferenceAdapter):
    model_type = "tts"
    estimated_vram_mb = 2000

    async def load(self, device: str) -> None:
        self._model = True

    async def infer(self, params: dict) -> InferenceResult:
        return InferenceResult(data=b"ok", content_type="text/plain")


def _make_spec(model_id="test-model", vram_mb=2000):
    return ModelSpec(id=model_id, model_type="tts", adapter_class="fake", path="/fake", vram_mb=vram_mb)


def _make_manager(specs=None):
    registry = MagicMock()
    registry.get = lambda mid: next((s for s in (specs or []) if s.id == mid), None)
    registry.specs = specs or []
    allocator = MagicMock()
    allocator.get_best_gpu = MagicMock(return_value=0)
    allocator.get_free_mb = MagicMock(return_value=20000)
    return ModelManager(registry=registry, allocator=allocator)


async def test_load_and_unload():
    mgr = _make_manager([_make_spec()])
    await mgr.load_model("test-model", adapter_factory=lambda sp: FakeAdapter(sp.path))
    assert mgr.is_loaded("test-model")
    await mgr.unload_model("test-model")
    assert not mgr.is_loaded("test-model")


async def test_load_unknown_model():
    mgr = _make_manager([])
    with pytest.raises(ValueError, match="Unknown model"):
        await mgr.load_model("nonexistent")


async def test_add_remove_reference():
    mgr = _make_manager([_make_spec()])
    await mgr.load_model("test-model", adapter_factory=lambda sp: FakeAdapter(sp.path))
    mgr.add_reference("test-model", "wf-1")
    assert mgr.get_references("test-model") == {"wf-1"}
    mgr.remove_reference("test-model", "wf-1")
    assert mgr.get_references("test-model") == set()


async def test_unload_skips_referenced():
    mgr = _make_manager([_make_spec()])
    await mgr.load_model("test-model", adapter_factory=lambda sp: FakeAdapter(sp.path))
    mgr.add_reference("test-model", "wf-1")
    await mgr.unload_model("test-model")
    assert mgr.is_loaded("test-model")  # still loaded


async def test_evict_lru():
    mgr = _make_manager([_make_spec("model-a"), _make_spec("model-b")])
    await mgr.load_model("model-a", adapter_factory=lambda sp: FakeAdapter(sp.path))
    await mgr.load_model("model-b", adapter_factory=lambda sp: FakeAdapter(sp.path))
    evicted = await mgr.evict_lru(gpu_index=0)
    assert evicted == "model-a"
    assert not mgr.is_loaded("model-a")
    assert mgr.is_loaded("model-b")


async def test_get_adapter():
    mgr = _make_manager([_make_spec()])
    await mgr.load_model("test-model", adapter_factory=lambda sp: FakeAdapter(sp.path))
    adapter = mgr.get_adapter("test-model")
    assert adapter is not None and adapter.is_loaded
