"""FakeTTSAdapter —— 零 GPU / 零真模型的 TTSEngine 实现。

V1.5 Lane F 用它跑通 TTS runner 子进程路径（IPC + 生命周期 + node_type=tts
分流）而不需要真硬件 / 真 TTS 权重。继承 src.workers.tts_engines.base.TTSEngine
（即 InferenceAdapter 的 TTS 子类），所以 runner / ModelManager 看它和真 TTS
adapter 形状完全一致。

不 import torch / torchaudio / soundfile —— synthesize 直接拼一段最小合法 WAV
header + 静音 PCM，conftest 的 CUDA_VISIBLE_DEVICES="" 对它无影响。

spec §4.4：TTS 节点只需 boundary-cancel。TTSEngine.infer(req) 签名只收 req，
不接 progress_callback / cancel_flag —— FakeTTSAdapter 保持这个形状。
"""
from __future__ import annotations

import struct
from typing import Any, ClassVar

from src.services.inference.base import MediaModality
from src.workers.tts_engines.base import TTSEngine, TTSResult


def _silent_wav(sample_rate: int, duration_seconds: float) -> bytes:
    """拼一段最小合法 WAV（PCM16 单声道静音）—— 不依赖 soundfile / torch。"""
    n_samples = max(1, int(sample_rate * duration_seconds))
    data = b"\x00\x00" * n_samples  # PCM16 静音
    byte_rate = sample_rate * 2
    block_align = 2
    riff_chunk_size = 36 + len(data)
    header = b"RIFF" + struct.pack("<I", riff_chunk_size) + b"WAVE"
    fmt = (
        b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
    )
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt + data_chunk


class FakeTTSAdapter(TTSEngine):
    """假 TTS adapter：load 无副作用，synthesize 返回固定静音 WAV。"""

    ENGINE_NAME: ClassVar[str] = "fake_tts"
    estimated_vram_mb: ClassVar[int] = 0
    modality = MediaModality.AUDIO

    def __init__(self, paths: dict[str, str], device: str = "cpu", **params: Any) -> None:
        super().__init__(paths=paths, device=device, **params)
        # fail_load 开关 —— 模拟权重丢失 / OOM，供 runner OOM/load-failed 路径测试
        self._fail_load = bool(params.get("fail_load", False))

    def load_sync(self) -> None:
        if self._fail_load:
            raise RuntimeError(f"fake tts load failure for paths={self.paths}")
        self._model = object()  # 非 None → is_loaded True

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("FakeTTSAdapter not loaded. Call load() first.")
        # 文本越长「时长」越长 —— 给 audio_seconds 一个可断言的非零值
        duration = round(max(0.1, len(text) * 0.05), 2)
        return TTSResult(
            audio_bytes=_silent_wav(sample_rate, duration),
            sample_rate=sample_rate,
            duration_seconds=duration,
            format="wav",
        )

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
