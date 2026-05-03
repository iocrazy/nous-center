from __future__ import annotations

import asyncio
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


class TTSResult(BaseModel):
    """Concrete-engine return shape from synthesize(). TTSEngine.infer wraps
    into the unified InferenceResult envelope before returning to callers."""

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"

    model_config = {"arbitrary_types_allowed": True}


class TTSEngine(InferenceAdapter):
    modality = MediaModality.AUDIO
    estimated_vram_mb = 0

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any):
        super().__init__(paths=paths, device=device)
        # All TTS engines are single-component: paths['main'] is the model dir.
        self.model_path = Path(paths.get("main", ""))

    async def load(self, device: str | None = None) -> None:
        if device:
            self.device = device
        # Wrap potentially-blocking model load in a thread so the event loop
        # stays responsive during long startup operations.
        await asyncio.to_thread(self.load_sync)

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

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        if not isinstance(req, AudioRequest):
            raise TypeError(f"TTSEngine expects AudioRequest, got {type(req).__name__}")
        import time

        t0 = time.monotonic()
        # synthesize() is sync (blocking torch); offload to thread so the
        # event loop stays responsive. Per-engine async-native rewrite is
        # explicitly out of scope (each engine owns blocking torch internals).
        result = await asyncio.to_thread(
            self.synthesize,
            text=req.text,
            voice=req.voice,
            speed=req.speed,
            sample_rate=req.sample_rate,
            reference_audio=req.reference_audio,
            reference_text=req.reference_text,
            emotion=req.emotion,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return InferenceResult(
            media_type=f"audio/{result.format}",
            data=result.audio_bytes,
            metadata={
                "sample_rate": result.sample_rate,
                "duration_seconds": result.duration_seconds,
                "format": result.format,
            },
            usage=UsageMeter(
                audio_seconds=result.duration_seconds,
                latency_ms=latency_ms,
            ),
        )

    def unload(self) -> None:
        self._model = None

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @property
    def supported_voices(self) -> list[str]:
        return ["default"]
