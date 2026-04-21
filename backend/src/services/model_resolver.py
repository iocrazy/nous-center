"""Resolve (api_key, request.model) → ServiceInstance.

Two regimes coexist:

1. Legacy 1:1 binding — `InstanceApiKey.instance_id` is NOT NULL.
   These keys existed before the M:N migration and continue to work
   unchanged: the bound instance wins and `request.model` is ignored.
   Customers who built against `{"model":"gpt-4"}` placeholder requests
   keep working without any server-side change.

2. M:N grants — `InstanceApiKey.instance_id` is NULL; the key owns
   rows in `ApiKeyGrant`. We look up active grants for this key and
   match `request.model` against `ServiceInstance.name`.

If neither regime produces a match we raise `ModelNotFound` with the
specific reason (no grants / unknown model / no model in request).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


class ModelNotFound(Exception):
    """No ServiceInstance matched the (api_key, model) pair."""


async def resolve_target_instance(
    session: AsyncSession,
    *,
    api_key: InstanceApiKey,
    requested_model: str | None,
) -> ServiceInstance:
    # Regime 1: legacy 1:1 binding wins unconditionally.
    if api_key.instance_id is not None:
        inst = await session.get(ServiceInstance, api_key.instance_id)
        if inst is None:
            raise ModelNotFound(
                f"legacy-bound instance {api_key.instance_id} disappeared",
            )
        return inst

    # Regime 2: M:N. Need a model name to pick.
    if not requested_model:
        raise ModelNotFound(
            "api key has no legacy binding; request.model is required",
        )

    stmt = (
        select(ServiceInstance)
        .join(ApiKeyGrant, ApiKeyGrant.instance_id == ServiceInstance.id)
        .where(
            ApiKeyGrant.api_key_id == api_key.id,
            ApiKeyGrant.status == "active",
            ServiceInstance.name == requested_model,
        )
    )
    inst = (await session.execute(stmt)).scalar_one_or_none()
    if inst is None:
        raise ModelNotFound(
            f"no active grant for model '{requested_model}' on this key",
        )
    return inst
