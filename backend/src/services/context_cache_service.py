"""Service layer for ContextCache: create, fetch (with expiry/scope checks),
hit-count + TTL extension, idempotent delete, expired sweep.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.errors import InvalidRequestError
from src.models.context_cache import ContextCache

logger = logging.getLogger(__name__)


def _new_cache_id() -> str:
    # token_urlsafe(12) produces exactly 16 base64url chars; PG PK constraint
    # catches the (astronomically rare) collision.
    return f"ctx-{secrets.token_urlsafe(12)}"


async def create_cache_row(
    session: AsyncSession,
    *,
    instance_id: int,
    api_key_id: int | None,
    model: str,
    messages: list[dict],
    prompt_tokens: int,
    ttl_seconds: int = 86400,
    mode: str = "common_prefix",
) -> ContextCache:
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError(
            "messages must be a non-empty list",
            param="messages",
            code="invalid_messages",
        )
    if not (60 <= ttl_seconds <= 604800):
        raise InvalidRequestError(
            "ttl out of range [60, 604800]",
            param="ttl",
            code="invalid_ttl",
        )

    now = datetime.now(timezone.utc)
    row = ContextCache(
        id=_new_cache_id(),
        instance_id=instance_id,
        api_key_id=api_key_id,
        model=model,
        mode=mode,
        messages_json=messages,
        prompt_tokens=prompt_tokens,
        ttl_seconds=ttl_seconds,
        expires_at=now + timedelta(seconds=ttl_seconds),
        hit_count=0,
        created_at=now,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def fetch_active_cache(
    session: AsyncSession,
    cache_id: str,
    instance_id: int,
) -> ContextCache | None:
    """Returns row only if exists, not expired, and belongs to instance."""
    now = datetime.now(timezone.utc)
    stmt = select(ContextCache).where(
        ContextCache.id == cache_id,
        ContextCache.instance_id == instance_id,
        ContextCache.expires_at > now,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def fetch_cache_any_instance(
    session: AsyncSession,
    cache_id: str,
) -> ContextCache | None:
    """Used by GET/DELETE handlers to distinguish 404 from 403 (wrong instance)."""
    return await session.get(ContextCache, cache_id)


async def increment_hit_and_extend(
    session: AsyncSession,
    cache_id: str,
    ttl_seconds: int,
    instance_id: int | None = None,
) -> None:
    """Atomic UPDATE: hit_count += 1, expires_at = now+ttl, last_used_at = now.

    Caller may pass instance_id for defense-in-depth scoping.
    """
    now = datetime.now(timezone.utc)
    new_exp = now + timedelta(seconds=ttl_seconds)
    stmt = update(ContextCache).where(ContextCache.id == cache_id)
    if instance_id is not None:
        stmt = stmt.where(ContextCache.instance_id == instance_id)
    stmt = stmt.values(
        hit_count=ContextCache.hit_count + 1,
        expires_at=new_exp,
        last_used_at=now,
    ).execution_options(synchronize_session="fetch")
    await session.execute(stmt)
    await session.commit()


async def delete_cache(
    session: AsyncSession,
    cache_id: str,
    instance_id: int,
) -> bool:
    """Idempotent delete; returns True if a row was actually removed."""
    stmt = (
        delete(ContextCache)
        .where(
            ContextCache.id == cache_id,
            ContextCache.instance_id == instance_id,
        )
        .execution_options(synchronize_session="fetch")
    )
    result = await session.execute(stmt)
    await session.commit()
    return (result.rowcount or 0) > 0


async def cleanup_expired(session: AsyncSession) -> int:
    """Delete all expired rows. Returns count."""
    now = datetime.now(timezone.utc)
    stmt = (
        delete(ContextCache)
        .where(ContextCache.expires_at < now)
        .execution_options(synchronize_session="fetch")
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount or 0
