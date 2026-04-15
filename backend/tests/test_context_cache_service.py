"""Unit tests for context_cache_service: TTL, scope, hit_count, cleanup."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import select

from src.errors import InvalidRequestError
from src.models.context_cache import ContextCache
from src.services.context_cache_service import (
    cleanup_expired,
    create_cache_row,
    delete_cache,
    fetch_active_cache,
    fetch_cache_any_instance,
    increment_hit_and_extend,
)


@pytest.mark.asyncio
async def test_create_and_fetch_roundtrip(db_session, sample_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="qwen3.5-35b",
        messages=[{"role": "system", "content": "hi"}],
        prompt_tokens=42,
        ttl_seconds=3600,
    )
    assert row.id.startswith("ctx-")
    assert len(row.id) == len("ctx-") + 16  # token_urlsafe(12) = 16 chars
    fetched = await fetch_active_cache(db_session, row.id, sample_instance.id)
    assert fetched is not None
    assert fetched.prompt_tokens == 42
    assert fetched.hit_count == 0
    assert fetched.last_used_at is None


@pytest.mark.asyncio
async def test_fetch_expired_returns_none(db_session, sample_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()
    assert await fetch_active_cache(db_session, row.id, sample_instance.id) is None


@pytest.mark.asyncio
async def test_wrong_instance_returns_none(db_session, sample_instance, other_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    assert await fetch_active_cache(db_session, row.id, other_instance.id) is None
    # but fetch_cache_any_instance still finds it (used for 403 vs 404 distinction)
    other_view = await fetch_cache_any_instance(db_session, row.id)
    assert other_view is not None
    assert other_view.instance_id == sample_instance.id


@pytest.mark.asyncio
async def test_hit_extends_and_counts(db_session, sample_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    original_exp = row.expires_at
    await increment_hit_and_extend(db_session, row.id, ttl_seconds=3600)
    refreshed = (
        await db_session.execute(select(ContextCache).where(ContextCache.id == row.id))
    ).scalar_one()
    assert refreshed.hit_count == 1
    assert refreshed.last_used_at is not None
    # SQLite drops tz info on roundtrip; compare as naive UTC timestamps
    new_naive = refreshed.expires_at.replace(tzinfo=None) if refreshed.expires_at.tzinfo \
        else refreshed.expires_at
    orig_naive = original_exp.replace(tzinfo=None) if original_exp.tzinfo \
        else original_exp
    assert new_naive >= orig_naive


@pytest.mark.asyncio
async def test_increment_with_instance_scope(db_session, sample_instance, other_instance):
    """Scoped UPDATE skips rows from other instances even if id matches."""
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    # try to increment via other instance scope — should be a no-op
    await increment_hit_and_extend(
        db_session, row.id, ttl_seconds=3600, instance_id=other_instance.id
    )
    refreshed = (
        await db_session.execute(select(ContextCache).where(ContextCache.id == row.id))
    ).scalar_one()
    assert refreshed.hit_count == 0


@pytest.mark.asyncio
async def test_delete_only_owned(db_session, sample_instance, other_instance):
    row = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    # other instance: returns False (no-op)
    assert await delete_cache(db_session, row.id, other_instance.id) is False
    # owner: returns True and row gone
    assert await delete_cache(db_session, row.id, sample_instance.id) is True
    assert await fetch_cache_any_instance(db_session, row.id) is None
    # second call: idempotent — no error, returns False
    assert await delete_cache(db_session, row.id, sample_instance.id) is False


@pytest.mark.asyncio
async def test_cleanup_deletes_only_expired(db_session, sample_instance):
    fresh = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    stale = await create_cache_row(
        db_session,
        instance_id=sample_instance.id,
        api_key_id=None,
        model="m",
        messages=[{"role": "system", "content": "x"}],
        prompt_tokens=1,
        ttl_seconds=3600,
    )
    stale.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    await db_session.commit()
    n = await cleanup_expired(db_session)
    assert n == 1
    assert await fetch_active_cache(db_session, fresh.id, sample_instance.id) is not None
    assert await fetch_cache_any_instance(db_session, stale.id) is None


@pytest.mark.asyncio
async def test_invalid_messages_raises(db_session, sample_instance):
    with pytest.raises(InvalidRequestError) as exc:
        await create_cache_row(
            db_session,
            instance_id=sample_instance.id,
            api_key_id=None,
            model="m",
            messages=[],
            prompt_tokens=0,
            ttl_seconds=3600,
        )
    assert exc.value.code == "invalid_messages"


@pytest.mark.asyncio
async def test_invalid_ttl_raises(db_session, sample_instance):
    with pytest.raises(InvalidRequestError) as exc:
        await create_cache_row(
            db_session,
            instance_id=sample_instance.id,
            api_key_id=None,
            model="m",
            messages=[{"role": "system", "content": "x"}],
            prompt_tokens=1,
            ttl_seconds=30,  # too low
        )
    assert exc.value.code == "invalid_ttl"
