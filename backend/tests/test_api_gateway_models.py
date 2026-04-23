"""Lane A: API gateway data model tests.

Covers the new ApiKeyGrant / ResourcePack / AlertRule tables, plus the
ServiceInstance.category/meter_dim additions and InstanceApiKey.instance_id
nullable migration. Concurrency tests for ResourcePack.consume live in
test_resource_pack.py (services/resource_pack.py's job, not pure model).
"""

from __future__ import annotations

import pytest

from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


@pytest.mark.asyncio
async def test_service_instance_has_category_and_meter_dim(db_session):
    inst = ServiceInstance(
        source_type="model",
        source_name="qwen",
        name="llm",
        type="llm",
        category="llm",
        meter_dim="tokens",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    assert inst.category == "llm"
    assert inst.meter_dim == "tokens"


@pytest.mark.asyncio
async def test_service_instance_category_and_meter_dim_nullable(db_session):
    """Old rows (pre-migration) have category=null + meter_dim=null."""
    inst = ServiceInstance(source_type="model", source_name="x", name="legacy", type="llm")
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    assert inst.category is None
    assert inst.meter_dim is None


@pytest.mark.asyncio
async def test_instance_api_key_instance_id_nullable(db_session):
    """New M:N keys have instance_id=null; grants table carries bindings."""
    key = InstanceApiKey(
        instance_id=None,  # <-- the new case
        label="mn-key",
        key_hash="x",
        key_prefix="sk-test-a",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    assert key.instance_id is None


@pytest.mark.asyncio
async def test_api_key_grant_unique_per_key_instance_pair(db_session, sample_instance):
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-t-1"
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    g1 = ApiKeyGrant(api_key_id=key.id, service_id=sample_instance.id)
    db_session.add(g1)
    await db_session.commit()

    # Second grant on the same (key, instance) must fail unique constraint.
    g2 = ApiKeyGrant(api_key_id=key.id, service_id=sample_instance.id)
    db_session.add(g2)
    with pytest.raises(Exception):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_api_key_grant_default_active(db_session, sample_instance):
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-t-2"
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    grant = ApiKeyGrant(api_key_id=key.id, service_id=sample_instance.id)
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)
    assert grant.status == "active"
    assert grant.activated_at is not None
    assert grant.paused_at is None
    assert grant.retired_at is None


@pytest.mark.asyncio
async def test_resource_pack_defaults(db_session, sample_instance):
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-t-3"
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    grant = ApiKeyGrant(api_key_id=key.id, service_id=sample_instance.id)
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)

    pack = ResourcePack(
        grant_id=grant.id,
        name="10k token trial",
        total_units=10_000,
    )
    db_session.add(pack)
    await db_session.commit()
    await db_session.refresh(pack)
    assert pack.used_units == 0
    assert pack.source == "purchased"
    assert pack.expires_at is None


@pytest.mark.asyncio
async def test_alert_rule_defaults_enabled(db_session, sample_instance):
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="x", key_prefix="sk-t-4"
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    grant = ApiKeyGrant(api_key_id=key.id, service_id=sample_instance.id)
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)

    rule = AlertRule(grant_id=grant.id, threshold_percent=80)
    db_session.add(rule)
    await db_session.commit()
    await db_session.refresh(rule)
    assert rule.enabled is True
    assert rule.last_notified_at is None
    assert rule.pack_id is None


def test_grant_cascade_declared_at_schema_level():
    """SQLite doesn't enforce FKs by default (dialect-dependent), so we check
    the ondelete clause is declared rather than runtime-delete. PG will
    enforce it in production."""
    pack_fk = [c for c in ResourcePack.__table__.foreign_keys if c.column.table.name == "api_key_grants"][0]
    rule_fk = [c for c in AlertRule.__table__.foreign_keys if c.column.table.name == "api_key_grants"][0]
    assert pack_fk.ondelete == "CASCADE"
    assert rule_fk.ondelete == "CASCADE"
