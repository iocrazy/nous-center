from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    """Unified return type for all inference adapters."""
    data: bytes
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceAdapter(ABC):
    """Abstract base for all model adapters (TTS, LLM, Image)."""

    model_type: str
    estimated_vram_mb: int

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    async def load(self, device: str) -> None:
        """Load model weights onto the given device."""

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None

    @abstractmethod
    async def infer(self, params: dict[str, Any]) -> InferenceResult:
        """Run inference with the given parameters."""
