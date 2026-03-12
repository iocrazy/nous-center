from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"


class TTSEngine(ABC):
    """Base class for all TTS engines."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory/GPU."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
    ) -> TTSResult:
        """Synthesize speech from text. Returns audio bytes."""

    def unload(self) -> None:
        """Release model from memory/GPU."""
        self._model = None

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique engine identifier."""

    @property
    def supported_voices(self) -> list[str]:
        """List of supported voice names."""
        return ["default"]
