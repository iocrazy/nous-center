"""OpenAI-compatible TTS endpoints."""

import base64
import io
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.config import load_model_configs

router = APIRouter(tags=["openai-compat"])


# --- /v1/audio/speech ---

class SpeechRequest(BaseModel):
    model: str = "cosyvoice2"
    input: str
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["wav", "mp3", "opus", "flac"] = "wav"


CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


@router.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """Generate audio from text (OpenAI TTS compatible)."""
    from src.workers.tts_engines import registry

    engine = registry._ENGINE_INSTANCES.get(req.model)
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            409,
            detail=f"Model '{req.model}' is not loaded. Load it first via POST /api/v1/engines/{req.model}/load",
        )

    try:
        result = engine.synthesize(
            text=req.input,
            voice=req.voice,
            speed=req.speed,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    audio_bytes = result.audio_bytes

    # Format conversion if needed (engine returns wav by default)
    if req.response_format != "wav" and req.response_format != result.format:
        try:
            audio_bytes = _convert_audio(audio_bytes, result.format, req.response_format, result.sample_rate)
        except Exception:
            # If conversion fails, return wav
            pass

    content_type = CONTENT_TYPE_MAP.get(req.response_format, "audio/wav")
    return Response(content=audio_bytes, media_type=content_type)


def _convert_audio(audio_bytes: bytes, src_fmt: str, dst_fmt: str, sample_rate: int) -> bytes:
    """Convert audio format using soundfile."""
    import numpy as np
    import soundfile as sf

    buf_in = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf_in, dtype="float32")

    buf_out = io.BytesIO()
    fmt_map = {"wav": "WAV", "flac": "FLAC", "opus": "OGG"}
    sf_fmt = fmt_map.get(dst_fmt)
    if sf_fmt is None:
        raise ValueError(f"Unsupported output format: {dst_fmt}")

    sf.write(buf_out, data, sr, format=sf_fmt)
    buf_out.seek(0)
    return buf_out.read()


# --- /v1/models ---

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "nous-center"
    type: str = "tts"
    status: str = "unloaded"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI compatible)."""
    from src.workers.tts_engines import registry

    configs = load_model_configs()
    data = []
    for key, cfg in configs.items():
        if cfg.get("type") != "tts":
            continue
        engine = registry._ENGINE_INSTANCES.get(key)
        is_loaded = engine is not None and engine.is_loaded
        data.append(ModelObject(
            id=key,
            type=cfg["type"],
            status="loaded" if is_loaded else "unloaded",
        ))
    return ModelListResponse(data=data)


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get a single model's info (OpenAI compatible)."""
    from src.workers.tts_engines import registry

    configs = load_model_configs()
    if model_id not in configs:
        raise HTTPException(404, detail=f"Model '{model_id}' not found")

    cfg = configs[model_id]
    engine = registry._ENGINE_INSTANCES.get(model_id)
    is_loaded = engine is not None and engine.is_loaded
    return ModelObject(
        id=model_id,
        type=cfg.get("type", "tts"),
        status="loaded" if is_loaded else "unloaded",
    )
