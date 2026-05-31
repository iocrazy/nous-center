"""round8:TTS 引擎 unload 释放 GPU —— 清模型属性 + empty_cache(同 #234 image 版)。"""
from unittest.mock import MagicMock

from src.workers.tts_engines.base import TTSEngine, TTSResult


class _FakeTTS(TTSEngine):
    ENGINE_NAME = "fake"

    def load_sync(self):
        self._model = object()

    def synthesize(self, text, voice="default", speed=1.0, sample_rate=24000,
                   reference_audio=None, reference_text=None, emotion=None):
        return TTSResult(audio_bytes=b"", sample_rate=sample_rate, duration_seconds=0.0)

    @property
    def engine_name(self):
        return self.ENGINE_NAME


def test_base_unload_clears_model_and_empty_cache(monkeypatch):
    e = _FakeTTS(paths={"main": "/m"}, device="cpu")
    e._model = object()
    assert e.is_loaded
    # 拦 torch.cuda.empty_cache 验证被调
    called = {"empty": 0}
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.empty_cache.side_effect = lambda: called.__setitem__("empty", called["empty"] + 1)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    e.unload()
    assert e._model is None
    assert not e.is_loaded
    assert called["empty"] == 1


def test_moss_unload_clears_processor():
    """moss override 清 _processor(GPU 上的 audio_tokenizer)。"""
    from src.workers.tts_engines.moss_tts import MOSSTTSEngine
    e = MOSSTTSEngine.__new__(MOSSTTSEngine)
    e._model = object()
    e._processor = object()
    e.unload()
    assert e._model is None and e._processor is None
