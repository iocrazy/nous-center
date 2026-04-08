from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from src.services.inference.base import InferenceAdapter, InferenceResult


@dataclass
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"


class TTSEngine(InferenceAdapter):
    model_type = "tts"
    estimated_vram_mb = 0

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(model_path=model_path, device=device)

    async def load(self, device: str | None = None) -> None:
        if device:
            self.device = device
        self.load_sync()

    def load_sync(self) -> None:
        raise NotImplementedError

    @abstractmethod
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
        ...

    async def infer(self, params: dict[str, Any]) -> InferenceResult:
        result = self.synthesize(
            text=params.get("text", ""),
            voice=params.get("voice", "default"),
            speed=params.get("speed", 1.0),
            sample_rate=params.get("sample_rate", 24000),
            reference_audio=params.get("reference_audio"),
            reference_text=params.get("reference_text"),
            emotion=params.get("emotion"),
        )
        return InferenceResult(
            data=result.audio_bytes,
            content_type="audio/wav",
            metadata={
                "sample_rate": result.sample_rate,
                "duration_seconds": result.duration_seconds,
                "format": result.format,
            },
        )

    def unload(self) -> None:
        self._model = None

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @property
    def supported_voices(self) -> list[str]:
        return ["default"]
