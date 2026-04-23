"""Lane D · AlertRule evaluation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.alert_rule import check_and_fire


async def _seed(db_session, *, total=1000, used=0):
    inst = ServiceInstance(
        source_type="model", source_name="x", name="svc", type="llm",
        category="llm", meter_dim="tokens",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-a",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    grant = ApiKeyGrant(api_key_id=key.id, service_id=inst.id)
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)
    pack = ResourcePack(
        grant_id=grant.id, name="p", total_units=total, used_units=used,
    )
    db_session.add(pack)
    await db_session.commit()
    await db_session.refresh(pack)
    return grant, pack


@pytest.mark.asyncio
async def test_fires_when_over_threshold(db_session):
    grant, _ = await _seed(db_session, total=1000, used=800)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=80))
    await db_session.commit()

    events = await check_and_fire(db_session, grant_id=grant.id)
    assert len(events) == 1
    assert events[0].threshold_percent == 80
    assert events[0].observed_percent == 80


@pytest.mark.asyncio
async def test_skips_when_under_threshold(db_session):
    grant, _ = await _seed(db_session, total=1000, used=500)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=80))
    await db_session.commit()
    events = await check_and_fire(db_session, grant_id=grant.id)
    assert events == []


@pytest.mark.asyncio
async def test_disabled_rule_never_fires(db_session):
    grant, _ = await _seed(db_session, total=1000, used=900)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=80, enabled=False))
    await db_session.commit()
    events = await check_and_fire(db_session, grant_id=grant.id)
    assert events == []


@pytest.mark.asyncio
async def test_debounce_within_24h(db_session):
    grant, _ = await _seed(db_session, total=1000, used=850)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=80))
    await db_session.commit()

    # First check fires.
    assert len(await check_and_fire(db_session, grant_id=grant.id)) == 1
    # Same-moment second check is deduped.
    assert await check_and_fire(db_session, grant_id=grant.id) == []
    # Simulate 25h later — fires again.
    future = datetime.now(timezone.utc) + timedelta(hours=25)
    assert len(await check_and_fire(db_session, grant_id=grant.id, now=future)) == 1


@pytest.mark.asyncio
async def test_pack_scoped_rule(db_session):
    grant, pack_a = await _seed(db_session, total=1000, used=200)
    # Add a second pack at 95%.
    pack_b = ResourcePack(
        grant_id=grant.id, name="b", total_units=100, used_units=95,
    )
    db_session.add(pack_b)
    # Rule scoped to pack_b only.
    await db_session.commit()
    await db_session.refresh(pack_b)
    db_session.add(
        AlertRule(grant_id=grant.id, threshold_percent=90, pack_id=pack_b.id)
    )
    await db_session.commit()

    events = await check_and_fire(db_session, grant_id=grant.id)
    assert len(events) == 1
    assert events[0].pack_id == pack_b.id
    assert events[0].observed_percent == 95


@pytest.mark.asyncio
async def test_aggregate_rule_across_packs(db_session):
    grant, _ = await _seed(db_session, total=1000, used=500)
    db_session.add(ResourcePack(
        grant_id=grant.id, name="b", total_units=1000, used_units=300,
    ))
    # Aggregate: 800/2000 = 40%
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=50))
    await db_session.commit()
    assert await check_and_fire(db_session, grant_id=grant.id) == []

    # Bump one pack to push aggregate over 50%.
    from sqlalchemy import update
    from src.models.api_gateway import ResourcePack as RP
    await db_session.execute(
        update(RP).where(RP.grant_id == grant.id, RP.name == "b").values(used_units=800)
    )
    await db_session.commit()
    events = await check_and_fire(db_session, grant_id=grant.id)
    assert len(events) == 1
    assert events[0].pack_id is None  # aggregate


@pytest.mark.asyncio
async def test_no_packs_no_events(db_session):
    inst = ServiceInstance(
        source_type="model", source_name="x", name="svc", type="llm",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-b",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    grant = ApiKeyGrant(api_key_id=key.id, service_id=inst.id)
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=50))
    await db_session.commit()

    # No packs -> aggregate 0/0 -> no fire.
    events = await check_and_fire(db_session, grant_id=grant.id)
    assert events == []
