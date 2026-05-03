from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceResult,
    MediaModality,
)
from src.workers.tts_engines.base import TTSEngine, TTSResult


class FakeTTSEngine(TTSEngine):
    ENGINE_NAME = "fake"
    def load_sync(self) -> None:
        self._model = "loaded"
    def synthesize(self, text, voice="default", speed=1.0, sample_rate=24000,
                   reference_audio=None, reference_text=None, emotion=None) -> TTSResult:
        return TTSResult(audio_bytes=b"fakewav", sample_rate=sample_rate, duration_seconds=1.0)
    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME


def test_tts_engine_is_inference_adapter():
    assert issubclass(TTSEngine, InferenceAdapter)


def test_tts_engine_modality():
    """v2: TTSEngine declares MediaModality.AUDIO."""
    assert TTSEngine.modality == MediaModality.AUDIO


async def test_tts_infer_delegates_to_synthesize(tmp_path):
    engine = FakeTTSEngine(paths={"main": str(tmp_path)}, device="cpu")
    engine.load_sync()
    req = AudioRequest(request_id="r1", text="hello")
    result = await engine.infer(req)
    assert isinstance(result, InferenceResult)
    assert result.media_type == "audio/wav"
    assert result.data == b"fakewav"
    assert result.usage.audio_seconds == 1.0
