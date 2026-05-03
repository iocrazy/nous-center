"""InferenceAdapter v2 — unified typed surface for all modalities.

PR-0 establishes the v2 contract; existing 7 adapters (vLLM/SGLang/5 TTS)
all migrate to this shape in the same commit. No v1 coexistence.

Concrete subclasses pin `modality` to a specific MediaModality, accept
`paths: dict[str, str]` for multi-component models (image: transformer +
text_encoder + vae; LLM/TTS: just `paths['main']`), implement
`infer(req)` with a typed Request subclass, and may override
`infer_stream(req)` for SSE/streaming.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Modality discriminator
# ------------------------------------------------------------------


class MediaModality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


# ------------------------------------------------------------------
# Typed request schemas (pydantic v2 discriminated union)
# ------------------------------------------------------------------


class InferenceRequest(BaseModel):
    """Base for typed inference requests.

    Subclasses MUST override `modality` as Literal[MediaModality.X]
    so pydantic's discriminated-union resolver dispatches JSON
    payloads to the correct subclass.
    """

    request_id: str = Field(..., description="Caller trace id")
    timeout_s: float | None = Field(None, gt=0)
    modality: MediaModality  # subclasses narrow to Literal[X]


class Message(BaseModel):
    """Multimodal-capable chat message (OpenAI chat completions schema)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class TextRequest(InferenceRequest):
    modality: Literal[MediaModality.TEXT] = MediaModality.TEXT
    messages: list[Message]
    model: str = ""  # opaque tag for upstream OpenAI-compat servers
    max_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = False
    enable_thinking: bool = False
    api_key: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class LoRASpec(BaseModel):
    """LoRA reference by display name (ComfyUI-style)."""

    name: str
    strength: float = Field(1.0, ge=-2, le=2)


class ImageRequest(InferenceRequest):
    modality: Literal[MediaModality.IMAGE] = MediaModality.IMAGE
    prompt: str
    negative_prompt: str = ""
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)
    steps: int = Field(25, ge=1, le=200)
    seed: int | None = None
    cfg_scale: float = Field(7.0, ge=0, le=30)
    loras: list[LoRASpec] = Field(default_factory=list)


class AudioRequest(InferenceRequest):
    modality: Literal[MediaModality.AUDIO] = MediaModality.AUDIO
    text: str
    voice: str = "default"
    speed: float = Field(1.0, gt=0, le=4)
    sample_rate: int = 24000
    reference_audio: str | None = None
    reference_text: str | None = None
    emotion: str | None = None
    format: Literal["wav", "mp3", "ogg"] = "wav"


class VideoRequest(InferenceRequest):
    """V0 schema-only placeholder. No backend implementation."""

    modality: Literal[MediaModality.VIDEO] = MediaModality.VIDEO
    prompt: str
    duration_s: float = Field(4.0, gt=0, le=30)
    fps: int = Field(24, ge=1, le=60)
    width: int = 1280
    height: int = 720


# ------------------------------------------------------------------
# Result envelope
# ------------------------------------------------------------------


class UsageMeter(BaseModel):
    """Cross-modality usage counter. Each adapter fills what applies."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    audio_seconds: float | None = None
    image_count: int | None = None
    video_seconds: float | None = None
    latency_ms: int


class StageTimings(BaseModel):
    """Stage-level timing for observability + post-mortem.

    Each adapter fills the fields that apply:
      LLM:   connect_ms, first_token_ms (TTFT), sample_ms, decode_ms
      Image: encode_ms, denoise_ms, vae_ms
      Audio: encode_ms, synthesize_ms
    """

    connect_ms: int | None = None
    first_token_ms: int | None = None
    encode_ms: int | None = None
    sample_ms: int | None = None
    decode_ms: int | None = None
    denoise_ms: int | None = None
    vae_ms: int | None = None
    synthesize_ms: int | None = None


class InferenceResult(BaseModel):
    """Unified result envelope across all modalities."""

    media_type: str  # "application/json" | "audio/wav" | "image/png" | ...
    data: bytes
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: UsageMeter

    model_config = {"arbitrary_types_allowed": True}


class StreamEvent(BaseModel):
    """Stream event for `infer_stream`. Flat envelope; payload is opaque."""

    type: Literal["progress", "delta", "done", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# ABC
# ------------------------------------------------------------------


class InferenceAdapter(ABC):
    """Adapter ABC.

    Concrete subclasses:
      - declare `modality: ClassVar[MediaModality]`
      - declare `estimated_vram_mb: ClassVar[int]`
      - implement `__init__(paths: dict[str, str], device: str = "cuda", **params)`
        — `paths['main']` is the primary file/dir for single-component models;
          image-class adapters read `paths['transformer']`, `paths['text_encoder']`,
          `paths['vae']`
      - implement `load(device)` and `infer(req)`
      - optionally override `infer_stream(req)` for SSE/streaming
        (presence detected via `supports_streaming()` classmethod —
         single source of truth, no separate flag to keep in sync)
    """

    modality: ClassVar[MediaModality] = MediaModality.MULTIMODAL
    estimated_vram_mb: ClassVar[int] = 0

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any):
        self.paths = paths
        self.device = device
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @classmethod
    def supports_streaming(cls) -> bool:
        """True iff subclass overrides `infer_stream`. Derived — no flag."""
        return cls.infer_stream is not InferenceAdapter.infer_stream

    @abstractmethod
    async def load(self, device: str) -> None:
        """Load model weights onto the given device."""

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None

    @abstractmethod
    async def infer(self, req: InferenceRequest) -> InferenceResult:
        """Run inference with a typed request."""

    async def infer_stream(
        self, req: InferenceRequest
    ) -> AsyncIterator[StreamEvent]:
        """Streaming inference. Default raises so non-streaming adapters
        signal "not supported" via `supports_streaming() == False`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer_stream"
        )
        if False:  # pragma: no cover  — satisfies AsyncIterator protocol
            yield  # type: ignore[unreachable]
