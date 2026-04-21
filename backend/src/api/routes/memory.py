"""Memory API (Wave 1): POST /sync + GET /prefetch."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.deps_auth import verify_bearer_token
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.memory.base import (
    MemoryProviderClientError,
    MemoryProviderInternalError,
)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
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
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
    instance, api_key = auth
    provider = request.app.state.memory_provider
    try:
        ids = await provider.add_entries(
            instance_id=instance.id,
            api_key_id=api_key.id if api_key else None,
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
    limit: int = 10,
    context_key: str | None = None,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
):
    instance, _ = auth
    provider = request.app.state.memory_provider
    results = await provider.prefetch(
        instance_id=instance.id,
        query=q,
        limit=limit,
        context_key=context_key,
    )
    return {"entries": results}
