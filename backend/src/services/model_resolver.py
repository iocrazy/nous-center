"""Resolve (api_key, request.model) → ServiceInstance.

v3 (2026-04-22): the Legacy 1:1 binding regime is gone. After the v3
migration every InstanceApiKey owns its bindings via ApiKeyGrant rows;
`InstanceApiKey.instance_id` is forced to NULL by the migration.

Resolution: look up active grants for this key and match `request.model`
against `ServiceInstance.name`. Raises `ModelNotFound` with a specific
reason (no model in request / no active grant for that name).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


class ModelNotFound(Exception):
    """No ServiceInstance matched the (api_key, model) pair."""


async def resolve_target_service(
    session: AsyncSession,
    *,
    api_key: InstanceApiKey,
    requested_model: str | None,
) -> ServiceInstance:
    if not requested_model:
        raise ModelNotFound("request.model is required")

    stmt = (
        select(ServiceInstance)
        .join(ApiKeyGrant, ApiKeyGrant.service_id == ServiceInstance.id)
        .where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
            ServiceInstance.name == requested_model,
        )
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise ModelNotFound(
            f"no active grant for service '{requested_model}' on this key",
        )
    return svc
