"""Lane F: TTS runner 迁入测试 —— FakeTTSAdapter 单测 + runner 子进程 TTS 路径。

零 GPU / 零真模型：FakeTTSAdapter 继承 TTSEngine 但 synthesize 返回固定 wav
bytes，不 import torch。runner 子进程测试起真 multiprocessing.Process。
"""
import uuid

import pytest

from src.runner.fake_tts_adapter import FakeTTSAdapter
from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceResult,
    MediaModality,
)
from src.workers.tts_engines.base import TTSEngine


def _audio_req(text: str = "你好世界") -> AudioRequest:
    return AudioRequest(request_id=str(uuid.uuid4()), text=text)


def test_fake_tts_adapter_is_inference_adapter():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    assert isinstance(a, InferenceAdapter)
    assert isinstance(a, TTSEngine)
    assert a.modality is MediaModality.AUDIO


@pytest.mark.asyncio
async def test_fake_tts_adapter_load_and_infer():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    assert a.is_loaded
    result = await a.infer(_audio_req())
    assert isinstance(result, InferenceResult)
    assert result.media_type == "audio/wav"
    assert result.data  # 非空 wav bytes
    assert result.metadata["sample_rate"] == 24000
    assert result.metadata["format"] == "wav"
    assert result.usage.audio_seconds is not None


@pytest.mark.asyncio
async def test_fake_tts_adapter_infer_before_load_raises():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    with pytest.raises(RuntimeError):
        await a.infer(_audio_req())


@pytest.mark.asyncio
async def test_fake_tts_adapter_rejects_non_audio_request():
    """TTSEngine.infer 对非 AudioRequest 抛 TypeError —— FakeTTSAdapter 继承此行为。"""
    from src.services.inference.base import ImageRequest

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    with pytest.raises(TypeError):
        await a.infer(ImageRequest(request_id="x", prompt="a cat"))
