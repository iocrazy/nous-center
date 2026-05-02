"""PR-0: v2 schema + ABC tests (pure v2, no v1 coexistence).

Covers:
- MediaModality enum + Literal discriminator on Request subclasses
- Pydantic validation (max_tokens has no upper ceiling, image dims clamped, etc.)
- supports_streaming() classmethod returns False by default,
  True when subclass overrides infer_stream
- InferenceAdapter requires `paths` kwarg in __init__
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from src.services.inference.base import (
    AudioRequest,
    ImageRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    LoRASpec,
    MediaModality,
    Message,
    StageTimings,
    StreamEvent,
    TextRequest,
    UsageMeter,
    VideoRequest,
)


# ------------------------------------------------------------------
# MediaModality + Request discriminator
# ------------------------------------------------------------------


def test_media_modality_values():
    assert MediaModality.TEXT.value == "text"
    assert MediaModality.IMAGE.value == "image"
    assert {m.value for m in MediaModality} == {
        "text", "audio", "image", "video", "embedding", "multimodal"
    }


def test_text_request_discriminator_locked():
    """Subclasses pin modality via Literal — assignment of wrong enum fails."""
    req = TextRequest(
        request_id="r1",
        messages=[Message(role="user", content="hi")],
    )
    assert req.modality == MediaModality.TEXT

    with pytest.raises(ValidationError):
        TextRequest(
            request_id="r1",
            messages=[Message(role="user", content="hi")],
            modality=MediaModality.IMAGE,  # type: ignore[arg-type]
        )


def test_image_request_clamps_dimensions():
    with pytest.raises(ValidationError):
        ImageRequest(request_id="r2", prompt="x", width=32)  # < 64
    with pytest.raises(ValidationError):
        ImageRequest(request_id="r2", prompt="x", height=8192)  # > 4096
    req = ImageRequest(request_id="r2", prompt="x", width=512, height=512, steps=10)
    assert req.modality == MediaModality.IMAGE
    assert req.steps == 10
    assert req.cfg_scale == 7.0
    assert req.loras == []


def test_text_request_max_tokens_no_upper_ceiling():
    """200k-context models must not be rejected at schema layer (per-model
    enforcement happens inside VLLMAdapter._clamp_max_tokens)."""
    req = TextRequest(
        request_id="r3",
        messages=[Message(role="user", content="x")],
        max_tokens=128_000 * 2,  # 256k — should NOT raise
    )
    assert req.max_tokens == 256_000


def test_text_request_supports_multimodal_messages():
    """`messages` content can be list[dict] for vision/audio."""
    req = TextRequest(
        request_id="r4",
        messages=[
            Message(role="system", content="be helpful"),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://..."}},
                ],
            ),
        ],
    )
    assert isinstance(req.messages[1].content, list)


def test_lora_spec_strength_range():
    LoRASpec(name="anime", strength=0.5)
    LoRASpec(name="anime", strength=-1.5)
    with pytest.raises(ValidationError):
        LoRASpec(name="anime", strength=3.0)


def test_audio_request_format_literal():
    req = AudioRequest(request_id="r5", text="hi", format="mp3")
    assert req.format == "mp3"
    with pytest.raises(ValidationError):
        AudioRequest(request_id="r5", text="hi", format="flac")  # type: ignore[arg-type]


def test_video_request_schema_only_placeholder():
    req = VideoRequest(request_id="r6", prompt="cat", duration_s=2.0)
    assert req.modality == MediaModality.VIDEO


def test_inference_request_requires_modality_field():
    """Base class can't be instantiated without modality."""
    with pytest.raises(ValidationError):
        InferenceRequest(request_id="r7")  # type: ignore[call-arg]


# ------------------------------------------------------------------
# Result envelope
# ------------------------------------------------------------------


def test_inference_result_envelope():
    res = InferenceResult(
        media_type="image/png",
        data=b"\x89PNG\r\n\x1a\n",
        usage=UsageMeter(latency_ms=14230, image_count=1),
        metadata={"actual_seed": 42, "stages": {"denoise_ms": 12500}},
    )
    assert res.media_type == "image/png"
    assert res.usage.image_count == 1
    assert res.metadata["actual_seed"] == 42


def test_stage_timings_all_optional():
    t_llm = StageTimings(connect_ms=12, first_token_ms=180, decode_ms=540)
    assert t_llm.denoise_ms is None
    t_image = StageTimings(encode_ms=1100, denoise_ms=12500, vae_ms=630)
    assert t_image.first_token_ms is None


def test_stream_event_types():
    StreamEvent(type="progress", payload={"step": 5})
    StreamEvent(type="done")
    with pytest.raises(ValidationError):
        StreamEvent(type="bogus")  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Adapter ABC: paths kwarg + supports_streaming derived
# ------------------------------------------------------------------


class _PlainAdapter(InferenceAdapter):
    modality = MediaModality.TEXT
    estimated_vram_mb = 100

    async def load(self, device: str) -> None:
        self._model = True

    async def infer(self, req):
        return InferenceResult(
            media_type="text/plain",
            data=b"ok",
            usage=UsageMeter(latency_ms=1),
        )


class _StreamingAdapter(InferenceAdapter):
    modality = MediaModality.TEXT
    estimated_vram_mb = 100

    async def load(self, device: str) -> None:
        self._model = True

    async def infer(self, req):
        return InferenceResult(
            media_type="text/plain",
            data=b"streamed",
            usage=UsageMeter(latency_ms=10),
        )

    async def infer_stream(self, req) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="delta", payload={"text": "hello"})
        yield StreamEvent(type="done")


def test_adapter_takes_paths_dict():
    adapter = _PlainAdapter(paths={"main": "/fake/model"})
    assert adapter.paths["main"] == "/fake/model"
    assert adapter.device == "cuda"


def test_adapter_supports_streaming_false_by_default():
    assert _PlainAdapter.supports_streaming() is False


def test_adapter_supports_streaming_derived_true():
    """Subclass that overrides infer_stream → reports True without a flag."""
    assert _StreamingAdapter.supports_streaming() is True


async def test_adapter_infer_typed():
    adapter = _PlainAdapter(paths={"main": "/fake"})
    await adapter.load("cpu")
    req = TextRequest(request_id="r", messages=[Message(role="user", content="x")])
    res = await adapter.infer(req)
    assert isinstance(res, InferenceResult)
    assert res.data == b"ok"


async def test_adapter_infer_stream_yields_events():
    adapter = _StreamingAdapter(paths={"main": "/fake"})
    req = TextRequest(request_id="r", messages=[Message(role="user", content="x")])
    events = [e async for e in adapter.infer_stream(req)]
    assert len(events) == 2
    assert events[0].type == "delta"
    assert events[1].type == "done"


async def test_adapter_default_infer_stream_raises():
    """Adapter that doesn't override infer_stream raises NotImplementedError."""
    adapter = _PlainAdapter(paths={"main": "/fake"})
    req = TextRequest(request_id="r", messages=[Message(role="user", content="x")])
    with pytest.raises(NotImplementedError):
        async for _ in adapter.infer_stream(req):
            pass
