"""Quota enforcement for the inference protocol layer.

Callers wrap each usage-recording path with `consume_for_request(...)`.
The function is deliberately small: it finds the active grant, charges
it, and fires any alerts. Nothing else.

Why alert checks swallow errors: a failure in the alert path (bad rule,
DB hiccup, misconfigured threshold) must never turn a successful
inference into a user-visible 500. The quota consume already committed
before we get here, so we log and move on.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_gateway import ApiKeyGrant
from src.services.alert_rule import AlertEvent, check_and_fire
from src.services.resource_pack import ConsumeResult, consume

logger = logging.getLogger(__name__)


class NoActiveGrant(Exception):
    """The key has no active grant for this instance — 403-shaped, not 402."""


async def consume_for_request(
    session: AsyncSession,
    *,
    api_key_id: int,
    instance_id: int,
    units: int,
) -> tuple[ConsumeResult, list[AlertEvent]]:
    """Atomically charge `units` against the grant for (api_key, instance).

    Returns (ConsumeResult, events). Raises:
      - NoActiveGrant: caller is not authorized; return 403.
      - QuotaExhausted (from resource_pack.consume): return 402.
    """
    grant_id = await session.scalar(
        select(ApiKeyGrant.id).where(
            ApiKeyGrant.api_key_id == api_key_id,
            ApiKeyGrant.instance_id == instance_id,
            ApiKeyGrant.status == "active",
        )
    )
    if grant_id is None:
        raise NoActiveGrant(
            f"api_key {api_key_id} has no active grant on instance {instance_id}",
        )

    result = await consume(session, grant_id=grant_id, units=units)

    # Best-effort: alert failures must not block the caller.
    try:
        events = await check_and_fire(session, grant_id=grant_id)
    except Exception:
        logger.exception("alert check failed for grant %s", grant_id)
        events = []

    return result, events
