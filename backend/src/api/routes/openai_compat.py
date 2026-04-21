"""OpenAI-compatible endpoints: chat/completions, audio/speech, models."""

import asyncio
import io
import json
import logging
import time
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.api.deps_auth import verify_bearer_token
from src.config import get_settings, load_model_configs
from src.errors import APIError, InvalidRequestError, NotFoundError, NousError
from src.models.service_instance import ServiceInstance
from src.models.instance_api_key import InstanceApiKey
from src.services.prompt_composer import (
    AgentLoadFailed,
    AgentNotFound,
    compose as compose_agent_prompt,
)
from src.services.skill_tools import skill_tool_schema

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai-compat"])


async def sse_with_error_envelope(inner):
    """Wrap an SSE async generator so any NousError/Exception is emitted as an
    OpenAI-style error chunk followed by exactly one `data: [DONE]`.

    - Strips stray `data: [DONE]` markers emitted by the inner generator so we
      always emit exactly one terminator from the wrapper.
    - Converts NousError via to_dict(); any other Exception becomes a generic
      APIError (no traceback leak).
    """
    try:
        async for chunk in inner:
            if chunk.strip() == "data: [DONE]":
                # wrapper owns the terminator
                continue
            yield chunk
    except NousError as e:
        yield f"data: {json.dumps(e.to_dict())}\n\n"
    except Exception:
        logger.exception("SSE stream failure")
        err = APIError("Internal server error", code="internal_error")
        yield f"data: {json.dumps(err.to_dict())}\n\n"
    finally:
        yield "data: [DONE]\n\n"


# --- thinking-mode model whitelist ---
# Models whose chat template honors `chat_template_kwargs.enable_thinking`.
# Match is by case-insensitive substring on the engine name. If a model is not
# listed, the `extra_body.thinking` field is silently ignored (per Step 2 spec
# decision C+A: whitelist with silent fallback).
_THINKING_MODEL_PATTERNS = (
    "qwen3",  # qwen3.5-35b, qwen3-8b, etc.
    "deepseek-r1",
    "deepseek-v3",
    "doubao-seed-1.8",
    "doubao-seed-2",
)


def _supports_thinking(engine_name: str) -> bool:
    n = (engine_name or "").lower()
    return any(p in n for p in _THINKING_MODEL_PATTERNS)


def _maybe_inject_thinking(body: dict, engine_name: str) -> None:
    """Translate `body['thinking'] = {'type': enabled|disabled|auto}` into
    `body['chat_template_kwargs']['enable_thinking'] = bool` for vLLM.

    - Pops `thinking` from body either way (vLLM rejects unknown top-level fields).
    - If model isn't whitelisted, silently drop (per Ark `extra_body` semantics:
      non-standard fields are best-effort, not hard contract).
    - `auto` = leave unset, let model default.
    """
    thinking = body.pop("thinking", None)
    if not isinstance(thinking, dict):
        return
    t = thinking.get("type")
    if t not in ("enabled", "disabled", "auto"):
        return
    if not _supports_thinking(engine_name):
        return
    if t == "auto":
        return
    kwargs = body.setdefault("chat_template_kwargs", {})
    kwargs["enable_thinking"] = (t == "enabled")


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

    # Resolve agent (top-level or extra_body.agent). vLLM rejects unknown
    # top-level fields, so always pop — even when injection is disabled.
    agent_id = body.pop("agent", None)
    if not agent_id and isinstance(body.get("extra_body"), dict):
        agent_id = body["extra_body"].pop("agent", None)
        if not body["extra_body"]:
            body.pop("extra_body", None)

    # Compose agent system message (chat/completions has no session concept,
    # so there's no binding check — every request is independent).
    settings = get_settings()
    agent_sys: str | None = None
    if settings.NOUS_ENABLE_AGENT_INJECTION and agent_id:
        try:
            agent_sys = compose_agent_prompt(agent_id, None)
        except AgentNotFound:
            raise InvalidRequestError(
                f"agent not found: {agent_id}",
                code="agent_not_found",
            )
        except AgentLoadFailed as e:
            logger.error("agent load failed: %s", e)
            raise APIError(
                f"failed to load agent {agent_id}",
                code="agent_load_failed",
            )

    if agent_sys is not None:
        messages = list(body.get("messages") or [])
        messages.insert(0, {"role": "system", "content": agent_sys})
        body["messages"] = messages
        # Inject Skill tool schema when an agent is active.
        tools_list = list(body.get("tools") or [])
        tools_list.insert(0, skill_tool_schema())
        body["tools"] = tools_list

    # Resolve context_id (top-level or extra_body.context_id)
    context_id = body.pop("context_id", None)
    if not context_id and isinstance(body.get("extra_body"), dict):
        context_id = body["extra_body"].pop("context_id", None)
        if not body["extra_body"]:
            body.pop("extra_body", None)

    if context_id:
        from src.models.database import create_session_factory as _csf
        from src.services.context_cache_service import (
            increment_hit_and_extend as _ihe,
            resolve_for_request,
        )

        sf = _csf()
        async with sf() as cache_session:
            cached_messages, cached_ttl = await resolve_for_request(
                cache_session,
                context_id=context_id,
                instance_id=instance.id,
                engine_name=engine_name,
            )
        if cached_messages:
            body["messages"] = cached_messages + list(body.get("messages", []))

        # Fire-and-forget hit-count update; loop persists across requests under uvicorn.
        async def _bump(cid: str = context_id, ttl: int = cached_ttl):
            try:
                async with _csf()() as s2:
                    await _ihe(s2, cid, ttl)
            except Exception:
                logger.exception("hit_count update failed for %s", cid)
        asyncio.create_task(_bump())

    # OpenAI SDK extra_body.thinking → vLLM chat_template_kwargs.enable_thinking
    # Whitelist-driven; silent ignore for unsupported models (Step 2 spec).
    _maybe_inject_thinking(body, engine_name)

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
                        error_text = (await resp.aread()).decode(errors="replace")
                        # Map upstream status to a NousError so the wrapper
                        # formats it uniformly.
                        if resp.status_code == 404:
                            raise NotFoundError(error_text[:500], code="upstream_not_found")
                        if 400 <= resp.status_code < 500:
                            raise InvalidRequestError(error_text[:500], code="upstream_bad_request")
                        raise APIError("Upstream LLM error", code="upstream_error")
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
                agent_id=agent_id if settings.NOUS_ENABLE_AGENT_INJECTION else None,
            )

        return StreamingResponse(
            sse_with_error_envelope(_stream_proxy()),
            media_type="text/event-stream",
        )

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
            agent_id=agent_id if settings.NOUS_ENABLE_AGENT_INJECTION else None,
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
async def create_speech(
    req: SpeechRequest,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
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
