from src.services.inference.base import InferenceAdapter, InferenceResult
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

async def test_tts_infer_delegates_to_synthesize(tmp_path):
    engine = FakeTTSEngine(model_path=str(tmp_path), device="cpu")
    engine.load_sync()
    result = await engine.infer({"text": "hello"})
    assert isinstance(result, InferenceResult)
    assert result.content_type == "audio/wav"
    assert result.data == b"fakewav"

async def test_tts_engine_model_type(tmp_path):
    engine = FakeTTSEngine(model_path=str(tmp_path), device="cpu")
    assert engine.model_type == "tts"
