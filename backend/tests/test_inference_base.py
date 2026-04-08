import pytest
from src.services.inference.base import InferenceAdapter, InferenceResult


class DummyAdapter(InferenceAdapter):
    model_type = "test"
    estimated_vram_mb = 100

    async def load(self, device: str) -> None:
        self._model = {"loaded": True}

    async def infer(self, params: dict) -> InferenceResult:
        return InferenceResult(data=b"ok", content_type="text/plain")


@pytest.fixture
def adapter(tmp_path):
    return DummyAdapter(model_path=str(tmp_path / "fake-model"), device="cpu")


async def test_adapter_lifecycle(adapter):
    assert not adapter.is_loaded
    await adapter.load("cpu")
    assert adapter.is_loaded
    result = await adapter.infer({})
    assert result.data == b"ok"
    assert result.content_type == "text/plain"
    adapter.unload()
    assert not adapter.is_loaded


async def test_inference_result_metadata():
    r = InferenceResult(data=b"wav", content_type="audio/wav", metadata={"duration": 1.5})
    assert r.metadata["duration"] == 1.5


async def test_adapter_default_device(tmp_path):
    a = DummyAdapter(model_path=str(tmp_path))
    assert a.device == "cuda"
    assert not a.is_loaded
