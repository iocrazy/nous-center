"""Responses API (Step 4) — POST / GET / LIST / DELETE.

Event-sourcing backed by response_sessions + response_turns. previous_response_id
stays in the public API for OpenAI SDK parity; internally maps to turn -> session
-> ORDER BY turn_idx assembly.

Streaming uses semantic SSE events (response.created / output_text.delta /
completed / error) compatible with OpenAI Python SDK's client.responses.stream().
A module-level background worker handles partial-write persistence on client
disconnect (await PG inside asyncio.CancelledError is unreliable).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token
from src.errors import (
    APIError, ConflictError, InvalidRequestError, NotFoundError,
    NousError, PermissionError as NousPermissionError,
)
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.response_session import ResponseSession, ResponseTurn
from src.models.service_instance import ServiceInstance
from src.services.context_cache_service import resolve_for_request
from src.services.responses_service import (
    SESSION_TOKEN_BUDGET,
    approx_tokens,
    assemble_history_for_response,
    check_session_budget,
    compact_messages,
    create_session,
    decode_content,
    update_session_usage,
    write_partial_assistant_turn,
    write_user_and_assistant_turns,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["responses"])


# ---------- tz normalization helper ---------- #

def _to_utc(dt: datetime | None) -> datetime | None:
    """SQLite stores tz-naive; PG returns tz-aware. Normalize for comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------- Partial-write background worker ---------- #

_partial_write_queue: asyncio.Queue | None = None


def _set_queue(q: asyncio.Queue | None) -> None:
    """Called from main.py lifespan to init/teardown the queue."""
    global _partial_write_queue
    _partial_write_queue = q


def schedule_partial_write(persist_fn, *args) -> None:
    """Fire-and-forget enqueue. Safe under request cancellation
    (put_nowait is sync, no await)."""
    if _partial_write_queue is None:
        logger.warning("partial_write_queue not initialized; drop")
        return
    try:
        _partial_write_queue.put_nowait((persist_fn, args))
    except asyncio.QueueFull:
        logger.error("partial-write queue full; drop")


async def partial_write_worker():
    """Drains partial-write requests serially. Started in lifespan.
    Survives request task cancellation."""
    assert _partial_write_queue is not None
    while True:
        item = await _partial_write_queue.get()
        if item is None:  # shutdown sentinel
            _partial_write_queue.task_done()
            break
        persist_fn, args = item
        try:
            await persist_fn(*args)
        except Exception:
            logger.exception("partial-write worker failed")
        finally:
            _partial_write_queue.task_done()


# ---------- SSE wrapper (semantic events) ---------- #

def _sse_format(evt: str, payload: dict) -> str:
    return f"event: {evt}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def responses_sse_envelope(inner, persist_partial_fn, request_id: str | None):
    """Wrap an async-iter of (evt_type, payload_dict) tuples, emit SSE wire format.

    Always emits exactly one `data: [DONE]\\n\\n` unless cancelled (in which case
    the socket is gone). Injects request_id into every event payload.

    Inner MUST yield `response.created` AFTER first vLLM byte (not at start),
    so failed requests see only `error` — never `created->error`.
    """
    accumulated_text = ""
    cancelled = False
    try:
        async for evt_type, payload in inner:
            payload = dict(payload)
            if request_id and "request_id" not in payload:
                payload["request_id"] = request_id
            if evt_type == "response.output_text.delta":
                accumulated_text += payload.get("delta", "")
            yield _sse_format(evt_type, payload)
    except asyncio.CancelledError:
        cancelled = True
        # schedule_partial_write is sync (put_nowait) — safe under cancellation.
        # await inside this block is unreliable; hand off to background worker.
        schedule_partial_write(
            persist_partial_fn,
            accumulated_text,
            "incomplete",
            "connection_closed",
        )
        raise
    except NousError as e:
        err_payload = {"type": "error", "error": e.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    except Exception:
        logger.exception("responses stream failure")
        err = APIError("Internal server error", code="internal_error")
        err_payload = {"type": "error", "error": err.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    finally:
        # Don't emit DONE on cancellation — socket is already gone.
        if not cancelled:
            yield "data: [DONE]\n\n"


# ---------- Input normalization + image dispatch ---------- #

def normalize_input(input_field: Any) -> list[dict]:
    """String -> [{role:user, content:[{type:input_text,text:...}]}];
    list-of-input-items -> wrap in user message;
    list-of-messages -> pass through."""
    if isinstance(input_field, str):
        return [{"role": "user", "content": [
            {"type": "input_text", "text": input_field}
        ]}]
    if isinstance(input_field, list):
        if input_field and all(
            isinstance(it, dict) and it.get("type", "").startswith("input_")
            for it in input_field
        ):
            return [{"role": "user", "content": input_field}]
        return input_field
    raise InvalidRequestError(
        "input must be string or array",
        param="input",
        code="invalid_input",
    )


def resolve_image(item: dict) -> dict:
    """input_image -> chat/completions image_url message content.
    file_id path reserved for Step 5 (Files API); raises 501."""
    if item.get("file_id"):
        err = NousError(
            "file_id input not supported until Step 5 (Files API)",
            code="image_file_id_not_implemented",
        )
        err.http_status = 501
        raise err
    if item.get("image_url"):
        return {
            "type": "image_url",
            "image_url": {
                "url": item["image_url"],
                "detail": item.get("detail", "auto"),
            },
        }
    raise InvalidRequestError(
        "input_image requires image_url or file_id",
        param="input_image",
        code="invalid_image_input",
    )


def transform_inputs_to_chat_messages(inputs: list[dict]) -> list[dict]:
    """Convert input items to chat/completions message format vLLM understands."""
    out = []
    for msg in inputs:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            transformed = []
            for item in content:
                t = item.get("type", "")
                if t == "input_text":
                    transformed.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                elif t == "input_image":
                    transformed.append(resolve_image(item))
                elif t == "output_text":
                    # Replaying a prior assistant turn
                    transformed.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                else:
                    transformed.append(item)  # pass-through; vLLM rejects if bad
            out.append({"role": role, "content": transformed})
    return out


# ---------- Request schema ---------- #

class _ThinkingCfg(BaseModel):
    type: str = "auto"


class _ReasoningCfg(BaseModel):
    effort: str = "medium"


class _TextFormatCfg(BaseModel):
    type: str = "text"
    json_schema: dict | None = None


class _TextCfg(BaseModel):
    format: _TextFormatCfg = Field(default_factory=_TextFormatCfg)


class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[Any]
    previous_response_id: str | None = None
    context_id: str | None = None
    instructions: str | None = None
    thinking: _ThinkingCfg = Field(default_factory=_ThinkingCfg)
    reasoning: _ReasoningCfg = Field(default_factory=_ReasoningCfg)
    store: bool = True
    expire_at: int | None = None
    stream: bool = False
    text: _TextCfg = Field(default_factory=_TextCfg)


# ---------- POST /v1/responses ---------- #

@router.post("/v1/responses")
async def create_response(
    req: CreateResponseRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth
    if instance.source_type != "model":
        raise InvalidRequestError(
            "Responses only supported on model-type instances",
            code="not_a_model_instance",
        )
    engine_name = instance.source_name or str(instance.source_id)

    # Adapter resolution
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is None:
        raise APIError("Model manager unavailable", code="model_manager_missing")
    adapter = model_mgr.get_adapter(engine_name)
    if adapter is None or not adapter.is_loaded:
        raise APIError(
            f"Model '{engine_name}' is not loaded",
            code="model_not_loaded",
        )
    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise APIError(
            "Model has no inference endpoint",
            code="no_inference_endpoint",
        )
    max_model_len = getattr(adapter, "max_model_len", 4096) or 4096

    # Step 3 cache resolution
    cached_messages, _ttl = await resolve_for_request(
        session,
        context_id=req.context_id,
        instance_id=instance.id,
        engine_name=engine_name,
    )

    # Previous-chain walk
    previous_messages: list[dict] = []
    sess: ResponseSession | None = None
    if req.previous_response_id:
        previous_messages, sess = await assemble_history_for_response(
            session, req.previous_response_id, instance_id=instance.id
        )
        if sess.model != engine_name:
            raise InvalidRequestError(
                f"Previous response was for '{sess.model}', not '{engine_name}'",
                code="previous_response_model_mismatch",
                param="model",
            )
        # Doc-convention warning if both context_id + previous_response_id
        if req.context_id:
            logger.warning(
                "both context_id=%s and previous_response_id=%s provided; "
                "chain already contains cache from first turn — skipping cache prepend",
                req.context_id, req.previous_response_id,
            )
            cached_messages = None

    # Normalize new input
    new_input_messages = transform_inputs_to_chat_messages(
        normalize_input(req.input)
    )

    # Assemble per MESSAGES_ORDER: context -> chain -> instructions -> input.
    # Both previous history and new input may contain API-facing types
    # (input_text / output_text) which vLLM doesn't understand — convert.
    previous_messages_vllm = transform_inputs_to_chat_messages(previous_messages)
    messages: list[dict] = []
    if cached_messages:
        messages.extend(cached_messages)
    messages.extend(previous_messages_vllm)
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    messages.extend(new_input_messages)

    # Compaction
    max_history_tokens = max_model_len - 2048
    compacted, history_truncated = compact_messages(
        messages, max_history_tokens=max_history_tokens
    )
    if approx_tokens(compacted) > max_history_tokens:
        raise InvalidRequestError(
            f"input alone exceeds max_history_tokens ({max_history_tokens})",
            code="input_too_long_for_model",
            param="input",
        )
    messages = compacted

    # Session budget check (only if continuing a session)
    estimated_input = approx_tokens(messages)
    if sess is not None:
        await check_session_budget(session, sess, estimated_new=estimated_input)
    else:
        # New session (not yet created; create now so writes have a target)
        sess = await create_session(
            session,
            instance_id=instance.id,
            api_key_id=api_key.id,
            model=engine_name,
            context_cache_id=req.context_id,
        )

    # Build vLLM body
    vllm_body: dict = {
        "model": "",
        "messages": messages,
        "max_tokens": 2048,  # TODO: read from req.max_output_tokens once added
        "stream": req.stream,
    }
    # Step 2: thinking mapping — mutates body in place, returns None.
    vllm_body["thinking"] = req.thinking.model_dump()
    from src.api.routes.openai_compat import _maybe_inject_thinking
    _maybe_inject_thinking(vllm_body, engine_name)
    # text.format passthrough
    if req.text.format.type == "json_schema" and req.text.format.json_schema:
        vllm_body["response_format"] = {
            "type": "json_schema",
            "json_schema": req.text.format.json_schema,
        }
    elif req.text.format.type == "json_object":
        vllm_body["response_format"] = {"type": "json_object"}

    request_id = getattr(request.state, "request_id", None)

    # Reject streaming in 6a; 6b adds the branch
    if req.stream:
        raise InvalidRequestError(
            "streaming not yet implemented",
            code="streaming_pending",
        )

    # ---- Non-streaming path ----
    async with httpx.AsyncClient(timeout=300, proxy=None) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=vllm_body,
        )
    if resp.status_code != 200:
        err_text = resp.text[:500]
        if 400 <= resp.status_code < 500:
            raise InvalidRequestError(err_text, code="upstream_bad_request")
        raise APIError("vLLM error", code="upstream_error")
    data = resp.json()
    choice = data["choices"][0]
    assistant_text = choice["message"].get("content") or ""
    finish_reason = choice.get("finish_reason", "stop")
    usage = data.get("usage", {}) or {}

    # Persist user + assistant turn pair
    user_content = (
        new_input_messages[-1]["content"] if new_input_messages else []
    )
    asst_content = [
        {"type": "output_text", "text": assistant_text, "annotations": []}
    ]
    status = "completed"
    incomplete_details = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}

    _, asst_turn = await write_user_and_assistant_turns(
        session,
        sess=sess,
        user_content=user_content,
        assistant_content=asst_content,
        usage={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
            },
        },
        reasoning=None,
        instructions=req.instructions,
        text_format=req.text.model_dump(),
        status=status,
        incomplete_reason=(incomplete_details or {}).get("reason"),
    )
    await update_session_usage(
        session, sess,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
    from src.services.usage_service import record_llm_usage
    await record_llm_usage(
        model=engine_name,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        duration_ms=0,
        instance_id=instance.id,
        api_key_id=api_key.id,
    )

    return {
        "id": asst_turn.id,
        "object": "response",
        "status": status,
        "incomplete_details": incomplete_details,
        "created_at": int(_to_utc(asst_turn.created_at).timestamp()),
        "model": engine_name,
        "previous_response_id": req.previous_response_id,
        "instructions": req.instructions,
        "store": req.store,
        "expire_at": int(_to_utc(sess.expire_at).timestamp()),
        "output": [
            {
                "type": "message",
                "id": f"msg-{asst_turn.id[5:]}",
                "role": "assistant",
                "content": asst_content,
            }
        ],
        "usage": asst_turn.usage_json,
        "history_truncated": history_truncated,
        "request_id": request_id,
    }
