"""Lane B-T2 · quota_gate tests.

quota_gate.consume_for_request is the thin wrapper the protocol layer
calls. It:

  1. Finds the active ApiKeyGrant for (api_key_id, instance_id). No grant
     means the key isn't authorized for this instance, NOT a quota issue.
  2. Calls resource_pack.consume on that grant. QuotaExhausted bubbles up.
  3. Fires alert_rule.check_and_fire after the successful consume. Alert
     evaluation is best-effort and must not block the caller — errors
     inside the alert check are logged and swallowed.

The gate returns (ConsumeResult, list[AlertEvent]).
"""

from __future__ import annotations

import pytest

from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.quota_gate import (
    NoActiveGrant,
    consume_for_request,
)
from src.services.resource_pack import QuotaExhausted


async def _make_kit(db_session, *, status: str = "active", pack_total: int = 1000):
    inst = ServiceInstance(
        source_type="model", source_name="qwen3", name="qwen3",
        type="llm", category="llm", meter_dim="tokens",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)

    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="h", key_prefix="sk-q",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    grant = ApiKeyGrant(
        api_key_id=key.id, service_id=inst.id, status=status,
    )
    db_session.add(grant)
    await db_session.commit()
    await db_session.refresh(grant)

    pack = ResourcePack(
        grant_id=grant.id, name="p", total_units=pack_total, used_units=0,
    )
    db_session.add(pack)
    await db_session.commit()
    await db_session.refresh(pack)
    return inst, key, grant, pack


@pytest.mark.asyncio
async def test_happy_path_consumes_and_returns(db_session):
    inst, key, _, _ = await _make_kit(db_session)
    result, events = await consume_for_request(
        db_session, api_key_id=key.id, service_id=inst.id, units=100,
    )
    assert result.remaining_units == 900
    assert events == []


@pytest.mark.asyncio
async def test_no_grant_raises(db_session):
    inst = ServiceInstance(
        source_type="model", source_name="x", name="x", type="llm",
    )
    db_session.add(inst)
    await db_session.commit()
    await db_session.refresh(inst)
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="h", key_prefix="sk-z",
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)

    with pytest.raises(NoActiveGrant):
        await consume_for_request(
            db_session, api_key_id=key.id, service_id=inst.id, units=1,
        )


@pytest.mark.asyncio
async def test_paused_grant_raises(db_session):
    inst, key, _, _ = await _make_kit(db_session, status="paused")
    with pytest.raises(NoActiveGrant):
        await consume_for_request(
            db_session, api_key_id=key.id, service_id=inst.id, units=1,
        )


@pytest.mark.asyncio
async def test_exhausted_pack_raises(db_session):
    inst, key, _, _ = await _make_kit(db_session, pack_total=10)
    with pytest.raises(QuotaExhausted):
        await consume_for_request(
            db_session, api_key_id=key.id, service_id=inst.id, units=100,
        )


@pytest.mark.asyncio
async def test_alert_fires_on_threshold_cross(db_session):
    inst, key, grant, pack = await _make_kit(db_session, pack_total=100)
    db_session.add(AlertRule(grant_id=grant.id, threshold_percent=50))
    await db_session.commit()

    _, events = await consume_for_request(
        db_session, api_key_id=key.id, service_id=inst.id, units=60,
    )
    assert len(events) == 1
    assert events[0].observed_percent == 60
