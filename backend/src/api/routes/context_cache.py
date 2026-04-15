"""Context Cache API routes: create / list / get / delete.

Pre-warm strategy: POST /v1/context/create calls vLLM once with max_tokens=1
so the actual prefix KV cache is hot before the user's first chat completion.
The vLLM call is NOT recorded in llm_usage (it's cache-warming overhead, not
billable inference).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token
from src.errors import (
    APIError,
    InvalidRequestError,
    NotFoundError,
    PermissionError as NousPermissionError,
)
from src.models.context_cache import ContextCache
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.context_cache_service import (
    create_cache_row,
    delete_cache,
    fetch_active_cache,
    fetch_cache_any_instance,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["context-cache"])


class CreateContextRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    ttl: int | None = Field(default=86400, ge=60, le=604800)


@router.post("/v1/context/create")
async def create_context(
    req: CreateContextRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth
    if instance.source_type != "model":
        raise InvalidRequestError(
            "Context Cache only supported on model-type instances",
            code="not_a_model_instance",
        )
    engine_name = instance.source_name or str(instance.source_id)

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
        raise APIError("Model has no inference endpoint", code="no_inference_endpoint")

    # Pre-warm: send messages through vLLM so prefix KV cache is hot, capture token count.
    # vLLM requires at least one user message; append a minimal one for the warm call only.
    # The prefix we actually cache is `req.messages` — vLLM's prefix matcher reuses the
    # KV for the system portion when the real chat comes in.
    has_user = any(
        isinstance(m, dict) and m.get("role") == "user" for m in req.messages
    )
    warm_messages = req.messages if has_user else (
        req.messages + [{"role": "user", "content": "."}]
    )
    warm_body = {
        "model": "",
        "messages": warm_messages,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60, proxy=None) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions", json=warm_body
            )
            if resp.status_code >= 400:
                err_text = resp.text[:500]
                if 400 <= resp.status_code < 500:
                    # 4xx = user's messages malformed; not a server error
                    raise InvalidRequestError(err_text, code="context_warm_rejected")
                raise APIError("vLLM pre-warm failed", code="warm_failed")
            warm_data = resp.json()
            prompt_tokens = warm_data.get("usage", {}).get("prompt_tokens", 0)
    except httpx.HTTPError as e:
        raise APIError(
            f"vLLM pre-warm transport error: {e}",
            code="warm_transport_failed",
        )

    row = await create_cache_row(
        session,
        instance_id=instance.id,
        api_key_id=api_key.id,
        model=engine_name,
        messages=req.messages,
        prompt_tokens=prompt_tokens,
        ttl_seconds=req.ttl or 86400,
    )

    return {
        "id": row.id,
        "model": row.model,
        "mode": row.mode,
        "ttl": row.ttl_seconds,
        "expires_at": row.expires_at.isoformat(),
        "usage": {
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": row.prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


@router.get("/v1/contexts")
async def list_contexts(
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    """List active (non-expired) caches for the caller's instance."""
    instance, _ = auth
    now = datetime.now(timezone.utc)
    stmt = (
        select(ContextCache)
        .where(
            ContextCache.instance_id == instance.id,
            ContextCache.expires_at > now,
        )
        .order_by(ContextCache.created_at.desc())
        .limit(200)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return {
        "data": [
            {
                "id": r.id,
                "model": r.model,
                "mode": r.mode,
                "ttl": r.ttl_seconds,
                "expires_at": r.expires_at.isoformat(),
                "created_at": r.created_at.isoformat(),
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
                "hit_count": r.hit_count,
                "prompt_tokens": r.prompt_tokens,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/v1/context/{cache_id}")
async def get_context(
    cache_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    row = await fetch_active_cache(session, cache_id, instance.id)
    if row is None:
        other = await fetch_cache_any_instance(session, cache_id)
        if other is not None and other.instance_id != instance.id:
            raise NousPermissionError(
                "Cache belongs to another instance",
                code="context_wrong_instance",
            )
        raise NotFoundError(
            "Context cache not found or expired",
            code="context_not_found",
        )

    preview = []
    for m in (row.messages_json or [])[:5]:
        c = m.get("content", "")
        if isinstance(c, str) and len(c) > 1000:
            c = c[:1000] + "..."
        preview.append({"role": m.get("role"), "content": c})

    return {
        "id": row.id,
        "model": row.model,
        "mode": row.mode,
        "ttl": row.ttl_seconds,
        "expires_at": row.expires_at.isoformat(),
        "created_at": row.created_at.isoformat(),
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "hit_count": row.hit_count,
        "prompt_tokens": row.prompt_tokens,
        "messages_preview": preview,
    }


@router.delete("/v1/context/{cache_id}", status_code=204)
async def delete_context(
    cache_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    other = await fetch_cache_any_instance(session, cache_id)
    if other is not None and other.instance_id != instance.id:
        raise NousPermissionError(
            "Cache belongs to another instance",
            code="context_wrong_instance",
        )
    # Idempotent — race between fetch_cache_any_instance and delete_cache returns
    # 204 even if the row was concurrently removed; that matches REST DELETE semantics.
    await delete_cache(session, cache_id, instance.id)
    return Response(status_code=204)
