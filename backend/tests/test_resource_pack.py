"""Lane D · ResourcePack.consume tests.

Covers the atomic CAS, expiry handling, multi-pack FIFO selection, and
a concurrency test that hammers a single pack with N concurrent consumers
to prove over-consumption is impossible.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.api_gateway import ApiKeyGrant, ResourcePack
from src.models.database import Base
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.resource_pack import QuotaExhausted, consume, peek_remaining


async def _make_grant_with_pack(
    session, *, total: int, used: int = 0, expires_at=None, source="purchased"
):
    inst = ServiceInstance(
        source_type="model", source_name="x", name="llm", type="llm",
        category="llm", meter_dim="tokens",
    )
    session.add(inst)
    await session.commit()
    await session.refresh(inst)

    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x",
        key_prefix=f"sk-{inst.id}",
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)

    grant = ApiKeyGrant(api_key_id=key.id, instance_id=inst.id)
    session.add(grant)
    await session.commit()
    await session.refresh(grant)

    pack = ResourcePack(
        grant_id=grant.id, name="p", total_units=total, used_units=used,
        expires_at=expires_at, source=source,
    )
    session.add(pack)
    await session.commit()
    await session.refresh(pack)
    return grant, pack


@pytest.mark.asyncio
async def test_consume_happy_path(db_session):
    grant, pack = await _make_grant_with_pack(db_session, total=1000)
    result = await consume(db_session, grant_id=grant.id, units=100)
    assert result.pack_id == pack.id
    assert result.remaining_units == 900


@pytest.mark.asyncio
async def test_consume_exact_boundary(db_session):
    """Consuming exactly what's left succeeds; one more unit fails."""
    grant, _ = await _make_grant_with_pack(db_session, total=100)
    result = await consume(db_session, grant_id=grant.id, units=100)
    assert result.remaining_units == 0
    with pytest.raises(QuotaExhausted):
        await consume(db_session, grant_id=grant.id, units=1)


@pytest.mark.asyncio
async def test_consume_insufficient(db_session):
    grant, _ = await _make_grant_with_pack(db_session, total=100, used=50)
    with pytest.raises(QuotaExhausted):
        await consume(db_session, grant_id=grant.id, units=80)


@pytest.mark.asyncio
async def test_consume_rejects_zero_and_negative(db_session):
    grant, _ = await _make_grant_with_pack(db_session, total=100)
    with pytest.raises(ValueError):
        await consume(db_session, grant_id=grant.id, units=0)
    with pytest.raises(ValueError):
        await consume(db_session, grant_id=grant.id, units=-5)


@pytest.mark.asyncio
async def test_consume_skips_expired_pack(db_session):
    past = datetime.now(timezone.utc) - timedelta(days=1)
    grant, _ = await _make_grant_with_pack(db_session, total=1000, expires_at=past)
    with pytest.raises(QuotaExhausted):
        await consume(db_session, grant_id=grant.id, units=10)


@pytest.mark.asyncio
async def test_consume_picks_soonest_expiry_first(db_session):
    """Multi-pack grant: FIFO by expiry; null expiry sorts last."""
    grant, p1 = await _make_grant_with_pack(db_session, total=100)
    # Add second pack with near-future expiry — should be preferred.
    soon = datetime.now(timezone.utc) + timedelta(hours=1)
    later = datetime.now(timezone.utc) + timedelta(days=30)
    p_no_expiry = ResourcePack(
        grant_id=grant.id, name="no-expire", total_units=100, expires_at=None,
    )
    p_late = ResourcePack(
        grant_id=grant.id, name="late", total_units=100, expires_at=later,
    )
    p_soon = ResourcePack(
        grant_id=grant.id, name="soon", total_units=100, expires_at=soon,
    )
    db_session.add_all([p_no_expiry, p_late, p_soon])
    await db_session.commit()
    await db_session.refresh(p_soon)

    # The "soon" pack wins.
    result = await consume(db_session, grant_id=grant.id, units=10)
    assert result.pack_id == p_soon.id


@pytest.mark.asyncio
async def test_peek_remaining(db_session):
    grant, _ = await _make_grant_with_pack(db_session, total=1000, used=200)
    # Add an expired pack — peek should ignore it.
    past = datetime.now(timezone.utc) - timedelta(days=1)
    db_session.add(ResourcePack(
        grant_id=grant.id, name="expired", total_units=500, used_units=0,
        expires_at=past,
    ))
    await db_session.commit()
    remaining = await peek_remaining(db_session, grant_id=grant.id)
    assert remaining == 800


@pytest.mark.asyncio
async def test_concurrent_consume_no_overcharge(tmp_path):
    """10 concurrent callers each try to consume 100 from a 500-unit pack.
    Exactly 5 succeed; used_units lands at 500 regardless of race order.

    Uses a file-backed SQLite (shared between independent sessions) so the
    atomicity is exercised rather than masked by SQLAlchemy's per-session
    identity map.
    """
    db_path = tmp_path / "concurrency.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    # Seed a grant + pack using one session.
    async with Session() as s:
        grant, pack = await _make_grant_with_pack(s, total=500)

    # Fire 10 concurrent consumers on fresh sessions.
    async def attempt(grant_id: int) -> bool:
        async with Session() as s:
            try:
                await consume(s, grant_id=grant_id, units=100)
                return True
            except QuotaExhausted:
                return False

    results = await asyncio.gather(*[attempt(grant.id) for _ in range(10)])
    succeeded = sum(1 for r in results if r)
    assert succeeded == 5, f"expected 5 successes, got {succeeded}"

    # Verify final state.
    async with Session() as s:
        final = await s.get(ResourcePack, pack.id)
        assert final.used_units == 500
        assert final.total_units - final.used_units == 0

    await engine.dispose()
