from abc import ABC, abstractmethod
from pathlib import Path


class LLMEngine(ABC):
    """Base class for LLM engines."""

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        self.model_path = Path(model_path)
        self.device = device
        self._config = kwargs
        self._process = None

    @property
    def is_loaded(self) -> bool:
        return self._process is not None

    @abstractmethod
    def load(self) -> None:
        """Start the LLM inference server."""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Generate text from prompt."""

    def unload(self) -> None:
        """Stop the inference server and release resources."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique engine identifier."""

    @property
    def base_url(self) -> str | None:
        """Return the base URL if the engine runs an HTTP server."""
        return None
