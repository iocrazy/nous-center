"""Memory API (Wave 1): POST /sync + GET /prefetch."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.deps_auth import verify_bearer_token_any
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.memory.base import (
    MemoryProviderClientError,
    MemoryProviderInternalError,
)

# legacy rip:外部 bearer API 必须挂 /v1(/api/* 被 AdminSessionGate 的浏览器 cookie 门挡死,
# M:N key 根本够不到 —— 同 #324 教训)。memory 按调用方 API key 切作用域。
router = APIRouter(prefix="/v1/memory", tags=["memory"])
logger = logging.getLogger(__name__)


class MemoryEntryIn(BaseModel):
    category: str = Field(..., pattern="^(preference|fact|instruction|custom)$")
    content: str
    context_key: str | None = None


class SyncRequest(BaseModel):
    entries: list[MemoryEntryIn]
    context_key: str | None = None


@router.post("/sync")
async def memory_sync(
    body: SyncRequest,
    request: Request,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
):
    _, api_key = auth
    provider = request.app.state.memory_provider
    try:
        ids = await provider.add_entries(
            owner_key_id=api_key.id,
            entries=[e.model_dump() for e in body.entries],
            context_key=body.context_key,
        )
        return {"entry_ids": ids}
    except MemoryProviderClientError as e:
        raise HTTPException(400, {"error": "invalid_entries", "message": str(e)})
    except MemoryProviderInternalError as e:
        logger.error("memory sync internal error: %s", e)
        raise HTTPException(500, {"error": "memory_backend_unavailable"})


@router.get("/prefetch")
async def memory_prefetch(
    request: Request,
    q: str = "",
    # round6:limit 加上界 —— 早先无约束直透 SQL LIMIT,`?limit=999999999` 拉整张
    # instance memory_entries 表(内存膨胀/慢查/响应体 DoS)。
    limit: int = Query(10, ge=1, le=100),
    context_key: str | None = None,
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(verify_bearer_token_any),
):
    _, api_key = auth
    provider = request.app.state.memory_provider
    results = await provider.prefetch(
        owner_key_id=api_key.id,
        query=q,
        limit=limit,
        context_key=context_key,
    )
    return {"entries": results}
