import pytest
from src.services.inference.base import (
    InferenceAdapter,
    InferenceResult,
    MediaModality,
    Message,
    TextRequest,
    UsageMeter,
)


class DummyAdapter(InferenceAdapter):
    modality = MediaModality.TEXT
    estimated_vram_mb = 100

    async def load(self, device: str) -> None:
        self._model = {"loaded": True}

    async def infer(self, req) -> InferenceResult:
        return InferenceResult(
            media_type="text/plain",
            data=b"ok",
            usage=UsageMeter(latency_ms=1),
        )


@pytest.fixture
def adapter(tmp_path):
    return DummyAdapter(paths={"main": str(tmp_path / "fake-model")}, device="cpu")


async def test_adapter_lifecycle(adapter):
    assert not adapter.is_loaded
    await adapter.load("cpu")
    assert adapter.is_loaded
    req = TextRequest(request_id="r", messages=[Message(role="user", content="x")])
    result = await adapter.infer(req)
    assert result.data == b"ok"
    assert result.media_type == "text/plain"
    adapter.unload()
    assert not adapter.is_loaded


async def test_inference_result_metadata():
    r = InferenceResult(
        media_type="audio/wav",
        data=b"wav",
        metadata={"duration": 1.5},
        usage=UsageMeter(audio_seconds=1.5, latency_ms=10),
    )
    assert r.metadata["duration"] == 1.5
    assert r.usage.audio_seconds == 1.5


async def test_adapter_default_device(tmp_path):
    a = DummyAdapter(paths={"main": str(tmp_path)})
    assert a.device == "cuda"
    assert not a.is_loaded
