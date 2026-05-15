"""Anthropic /v1/messages 兼容端点 — 转译到内部 vLLM chat completions。

m10 起，每个 service 暴露三种调用路径（OpenAI / Ollama / Anthropic），
本模块负责 Anthropic Messages API 这一档。SDK 期望：

  POST /v1/messages
  Headers:
    x-api-key: sk-...        (优先) 或 Authorization: Bearer sk-...
    anthropic-version: 2023-06-01
  Body:
    {
      "model": "<service-name>",
      "system": "...optional system prompt...",
      "messages": [{"role": "user", "content": "hi"}, ...],
      "max_tokens": 1024,
      "temperature": 0.7,
      "stream": false
    }

非流式响应：
    {
      "id": "msg_...", "type": "message", "role": "assistant",
      "model": "<engine>",
      "content": [{"type": "text", "text": "..."}],
      "stop_reason": "end_turn",
      "usage": {"input_tokens": N, "output_tokens": M}
    }

设计决策：
* 非流式直接对 openai_compat 的 dispatch 逻辑做就地复制（独立闭环，
  避免跨 module 的引用回环）。代码~80 行可控。
* 流式：本版返回 400 并提示尚未支持，留 TODO 给后续 PR 把 OpenAI SSE
  chunks 翻成 Anthropic event 流。
* 鉴权：复用 verify_bearer_token_any，但额外允许 `x-api-key` 头（这是
  Anthropic SDK 的官方写法）。
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token_any
from src.errors import APIError, InvalidRequestError, NotFoundError
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.inference.vllm_endpoint import (
    VLLMNoEndpoint,
    VLLMNotLoaded,
    get_vllm_base_url,
)
from src.services.model_resolver import ModelNotFound, resolve_target_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["anthropic-compat"])


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic 支持 content 是 string 或 [{"type":"text","text":"..."}, ...]
    content: str | list[dict]


class AnthropicRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[AnthropicMessage]
    system: str | None = None
    max_tokens: int = Field(1024, ge=1, le=200_000)
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False


def _flatten_content(content: str | list[dict]) -> str:
    """把 Anthropic content（str 或 blocks）压成单段 string，给 OpenAI 用。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _to_openai_messages(req: AnthropicRequest) -> list[dict]:
    msgs: list[dict] = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    for m in req.messages:
        msgs.append({"role": m.role, "content": _flatten_content(m.content)})
    return msgs


async def _verify_anthropic_key(
    x_api_key: str | None = Header(None, alias="x-api-key"),
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey]:
    """Anthropic SDK 用 x-api-key；我们也接受 Bearer 兜底。

    内部统一拼回 `Authorization: Bearer <key>` 复用 verify_bearer_token_any
    （它把 Header(...) 当默认值，显式 kwarg 会覆盖）。
    """
    if x_api_key:
        return await verify_bearer_token_any(
            authorization=f"Bearer {x_api_key}", session=session,
        )
    if authorization:
        return await verify_bearer_token_any(
            authorization=authorization, session=session,
        )
    raise HTTPException(401, detail="Missing x-api-key or Authorization header")


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    body: AnthropicRequest,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(
        _verify_anthropic_key,
    ),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth

    if body.stream:
        # TODO m10+: 翻 OpenAI SSE chunks 成 Anthropic event 流
        # (message_start / content_block_delta / message_stop)
        raise HTTPException(
            400,
            detail="streaming not implemented for /v1/messages yet; "
                   "use stream=false or hit /v1/chat/completions instead",
        )

    # Resolve target service —— 路径与 openai_compat 一致。
    if instance is None:
        try:
            instance = await resolve_target_service(
                session, api_key=api_key, requested_model=body.model,
            )
        except ModelNotFound as e:
            raise NotFoundError(str(e), code="model_not_found")
        if instance.status != "active":
            raise HTTPException(403, detail="Instance is inactive")
        await session.refresh(
            instance,
            attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
        )

    if instance.source_type != "model":
        # workflow / app dispatch 还没实现（与 openai_compat 一致）。
        raise HTTPException(
            501,
            detail=f"/v1/messages 暂不支持 {instance.source_type} 类型 service",
        )

    engine_name = instance.source_name or str(instance.source_id)
    # spec §4.5 D6/D8: direct-to-vLLM HTTP. base-URL lookup via single source of truth.
    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise HTTPException(503, detail=str(e)) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e

    openai_body: dict = {
        "model": "",  # vLLM 用自己的 model path
        "messages": _to_openai_messages(body),
        "max_tokens": body.max_tokens,
    }
    if body.temperature is not None:
        openai_body["temperature"] = body.temperature
    if body.top_p is not None:
        openai_body["top_p"] = body.top_p
    if body.stop_sequences:
        openai_body["stop"] = body.stop_sequences

    start_ms = time.monotonic()
    async with httpx.AsyncClient(timeout=300, proxy=None) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions", json=openai_body,
        )
    duration = int((time.monotonic() - start_ms) * 1000)

    if resp.status_code != 200:
        text = resp.text[:500]
        if resp.status_code == 404:
            raise NotFoundError(text, code="upstream_not_found")
        if 400 <= resp.status_code < 500:
            raise InvalidRequestError(text, code="upstream_bad_request")
        raise APIError("Upstream LLM error", code="upstream_error")

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    finish_reason = choice.get("finish_reason") or "end_turn"
    usage = data.get("usage", {})

    # 记录用量（与 openai_compat 一致）。
    from src.services.usage_service import record_llm_usage
    await record_llm_usage(
        model=engine_name,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        duration_ms=duration,
        instance_id=instance.id,
        api_key_id=api_key.id,
        agent_id=None,
    )

    # 反向 finish_reason：OpenAI "stop" → Anthropic "end_turn"
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, finish_reason)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": engine_name,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
