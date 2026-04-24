"""m04 dashboard summary — happy path on SQLite.

Stays away from PG-specific functions (the query uses plain SUM/COUNT
over today/yesterday windows) so the test suite still exercises the
shape end to end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.llm_usage import LLMUsage
from src.models.service_instance import ServiceInstance
from src.models.tts_usage import TTSUsage


@pytest.fixture
async def seeded(db_session):
    """Seeds 3 today-LLM + 1 yesterday-LLM + 1 today-TTS plus 3 keys
    (one bound, one orphan, one with active grant). Uses small offsets
    from `now` so the window math doesn't trip on the start-of-day
    boundary regardless of when the test runs."""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    one_min = timedelta(minutes=1)

    svc_a = ServiceInstance(
        name="svc-a", type="inference", status="active",
        source_type="model", source_name="qwen", category="llm", meter_dim="tokens",
    )
    svc_b = ServiceInstance(
        name="svc-b", type="inference", status="active",
        source_type="model", source_name="claude", category="llm", meter_dim="tokens",
    )
    db_session.add_all([svc_a, svc_b])
    await db_session.flush()

    key_bound = InstanceApiKey(
        instance_id=svc_a.id, label="legacy", key_hash="h", key_prefix="sk-leg",
    )
    key_unbound = InstanceApiKey(
        instance_id=None, label="orphan", key_hash="h", key_prefix="sk-orp",
    )
    key_with_grant = InstanceApiKey(
        instance_id=None, label="active-mn", key_hash="h", key_prefix="sk-mn",
    )
    db_session.add_all([key_bound, key_unbound, key_with_grant])
    await db_session.flush()

    grant = ApiKeyGrant(
        api_key_id=key_with_grant.id, service_id=svc_a.id, status="active",
    )
    db_session.add(grant)
    await db_session.flush()

    pack = ResourcePack(
        grant_id=grant.id, name="100K tokens",
        total_units=100_000, used_units=0,
    )
    db_session.add(pack)
    await db_session.flush()

    db_session.add(
        AlertRule(
            grant_id=grant.id,
            threshold_percent=82,
            enabled=True,
            last_notified_at=now - timedelta(minutes=5),
        )
    )

    db_session.add_all([
        LLMUsage(
            instance_id=svc_a.id, api_key_id=key_with_grant.id,
            model="qwen", prompt_tokens=10, completion_tokens=20, total_tokens=30,
            duration_ms=100, created_at=now - one_min,
        ),
        LLMUsage(
            instance_id=svc_a.id, api_key_id=key_with_grant.id,
            model="qwen", prompt_tokens=5, completion_tokens=15, total_tokens=20,
            duration_ms=150, created_at=now - 2 * one_min,
        ),
        LLMUsage(
            instance_id=svc_b.id, api_key_id=key_with_grant.id,
            model="claude", prompt_tokens=100, completion_tokens=200, total_tokens=300,
            duration_ms=200, created_at=now - 3 * one_min,
        ),
        # yesterday — clearly before today_start
        LLMUsage(
            instance_id=svc_a.id, api_key_id=key_with_grant.id,
            model="qwen", prompt_tokens=1, completion_tokens=2, total_tokens=3,
            duration_ms=50, created_at=today_start - timedelta(hours=2),
        ),
        TTSUsage(
            engine="cosyvoice", characters=10, duration_ms=80,
            created_at=now - 4 * one_min,
        ),
    ])
    await db_session.commit()
    return locals()


@pytest.mark.asyncio
async def test_summary_returns_business_aggregates(db_client, seeded):
    r = await db_client.get("/api/v1/dashboard/summary")
    assert r.status_code == 200, r.text
    data = r.json()
    # 3 LLM today + 1 TTS today
    assert data["today_calls"] == 4
    # delta_pct: today=4, yesterday=1 → 300%
    assert data["today_calls_delta_pct"] == pytest.approx(300.0, abs=0.1)
    assert data["month_tokens"] == 353
    assert data["month_tokens_quota"] == 100_000
    assert data["month_tokens_used_pct"] == pytest.approx(0.4, abs=0.1)
    assert data["service_count"] == 2
    assert data["api_key_count"] == 3
    # one truly unbound (no instance_id, no active grant)
    assert data["unbound_key_count"] == 1


@pytest.mark.asyncio
async def test_summary_top_services_today(db_client, seeded):
    r = await db_client.get("/api/v1/dashboard/summary")
    assert r.status_code == 200
    rows = r.json()["top_services_today"]
    assert [r["service_name"] for r in rows] == ["svc-a", "svc-b"]
    # svc-a: 2/3 LLM calls today = 66.7%
    assert rows[0]["calls"] == 2
    assert rows[0]["percent"] == pytest.approx(66.7, abs=0.1)
    assert rows[1]["calls"] == 1


@pytest.mark.asyncio
async def test_summary_recent_alerts(db_client, seeded):
    r = await db_client.get("/api/v1/dashboard/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["active_alerts_count"] == 1
    assert data["active_alerts_top_label"] == "svc-a"
    assert len(data["recent_alerts"]) == 1
    a = data["recent_alerts"][0]
    assert a["service_name"] == "svc-a"
    assert a["threshold_percent"] == 82
    assert a["severity"] == "err"  # ≥80 → err


@pytest.mark.asyncio
async def test_summary_empty_instance(db_client):
    r = await db_client.get("/api/v1/dashboard/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["today_calls"] == 0
    assert data["today_calls_delta_pct"] is None
    assert data["month_tokens_quota"] is None
    assert data["recent_alerts"] == []
    assert data["top_services_today"] == []
