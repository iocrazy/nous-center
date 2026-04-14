"""OpenAI-compatible endpoints: chat/completions, audio/speech, models."""

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.api.deps_auth import verify_bearer_token
from src.config import load_model_configs
from src.models.service_instance import ServiceInstance
from src.models.instance_api_key import InstanceApiKey

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compat"])


# --- /v1/chat/completions ---

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
    """OpenAI-compatible chat completions with token metering."""
    instance, api_key = auth

    # Resolve engine from instance
    if instance.source_type != "model":
        raise HTTPException(400, detail="This endpoint only supports model-type instances")

    engine_name = instance.source_name or str(instance.source_id)
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is None:
        raise HTTPException(500, detail="Model manager not available")

    adapter = model_mgr.get_adapter(engine_name)
    if adapter is None or not adapter.is_loaded:
        raise HTTPException(503, detail=f"Model '{engine_name}' is not loaded. Load it from the management page.")

    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise HTTPException(500, detail="Model has no inference endpoint")

    # Parse request body
    body = await request.json()
    body["model"] = ""  # vLLM uses its own model path

    # Clamp max_tokens
    max_model_len = getattr(adapter, "max_model_len", 4096) or 4096
    if body.get("max_tokens") and body["max_tokens"] > max_model_len - 512:
        body["max_tokens"] = max(max_model_len - 512, max_model_len // 2)

    is_stream = body.get("stream", False)
    start_ms = time.monotonic()

    if is_stream:
        # Streaming: inject include_usage, proxy SSE chunks
        body.setdefault("stream_options", {})["include_usage"] = True

        async def _stream_proxy():
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            async with httpx.AsyncClient(timeout=300, proxy=None) as client:
                async with client.stream(
                    "POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield f"data: {error_body.decode()}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        yield line + "\n"
                        # Extract usage from final chunk
                        if line.startswith("data: ") and line[6:] != "[DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if "usage" in chunk and chunk["usage"]:
                                    usage = chunk["usage"]
                            except Exception:
                                pass
                    yield "\n"

            # Record usage after stream completes
            duration = int((time.monotonic() - start_ms) * 1000)
            from src.services.usage_service import record_llm_usage
            await record_llm_usage(
                model=engine_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                duration_ms=duration,
                instance_id=instance.id,
                api_key_id=api_key.id,
            )

        return StreamingResponse(_stream_proxy(), media_type="text/event-stream")

    else:
        # Non-streaming: proxy request, extract usage
        async with httpx.AsyncClient(timeout=300, proxy=None) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body)

        duration = int((time.monotonic() - start_ms) * 1000)

        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

        data = resp.json()
        usage = data.get("usage", {})

        # Record usage
        from src.services.usage_service import record_llm_usage
        await record_llm_usage(
            model=engine_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_ms=duration,
            instance_id=instance.id,
            api_key_id=api_key.id,
        )

        return Response(content=resp.content, media_type="application/json")


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
async def list_models(request: Request):
    """List available models (OpenAI compatible)."""
    from src.workers.tts_engines import registry

    configs = load_model_configs()
    model_mgr = getattr(request.app.state, "model_manager", None)
    data = []
    for key, cfg in configs.items():
        model_type = cfg.get("type", "")
        if model_type == "tts":
            engine = registry._ENGINE_INSTANCES.get(key)
            is_loaded = engine is not None and engine.is_loaded
        elif model_type == "llm" and model_mgr:
            is_loaded = model_mgr.is_loaded(key)
        else:
            continue
        data.append(ModelObject(
            id=key,
            type=model_type,
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
