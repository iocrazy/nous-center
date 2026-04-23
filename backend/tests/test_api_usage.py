"""m13 usage routes — happy-path shape tests on SQLite.

Stays away from PG-specific aggregates (no percentile_cont, no
date_trunc — the route falls back to strftime under SQLite). We seed a
handful of LLM/TTS rows and assert each endpoint returns the documented
shape with reasonable values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models.instance_api_key import InstanceApiKey
from src.models.llm_usage import LLMUsage
from src.models.service_instance import ServiceInstance
from src.models.tts_usage import TTSUsage


def _admin_headers() -> dict[str, str]:
    return {}


@pytest.fixture
async def seeded(db_session):
    """Seed two services + a couple of api keys + 5 usage rows."""
    now = datetime.now(timezone.utc)

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

    key_legacy = InstanceApiKey(
        instance_id=svc_a.id, label="legacy-key", key_hash="h", key_prefix="sk-leg",
    )
    key_mn = InstanceApiKey(
        instance_id=None, label="mn-key", key_hash="h", key_prefix="sk-mn",
    )
    db_session.add_all([key_legacy, key_mn])
    await db_session.flush()

    db_session.add_all([
        LLMUsage(
            instance_id=svc_a.id, api_key_id=key_legacy.id,
            model="qwen", prompt_tokens=10, completion_tokens=20, total_tokens=30,
            duration_ms=100, created_at=now - timedelta(hours=1),
        ),
        LLMUsage(
            instance_id=svc_a.id, api_key_id=key_mn.id,
            model="qwen", prompt_tokens=5, completion_tokens=15, total_tokens=20,
            duration_ms=150, created_at=now - timedelta(hours=2),
        ),
        LLMUsage(
            instance_id=svc_b.id, api_key_id=key_mn.id,
            model="claude", prompt_tokens=100, completion_tokens=200, total_tokens=300,
            duration_ms=200, created_at=now - timedelta(days=1),
        ),
        TTSUsage(
            engine="cosyvoice", characters=42, duration_ms=80,
            created_at=now - timedelta(hours=3),
        ),
    ])
    await db_session.commit()
    return {"svc_a": svc_a, "svc_b": svc_b, "key_legacy": key_legacy, "key_mn": key_mn}


@pytest.mark.asyncio
async def test_summary_returns_aggregate_fields(db_client, seeded):
    r = await db_client.get("/api/v1/usage/summary?days=7", headers=_admin_headers())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["days"] == 7
    # 3 LLM + 1 TTS
    assert data["total_calls"] == 4
    assert data["total_tokens"] == 350
    assert data["prompt_tokens"] == 115
    assert data["completion_tokens"] == 235
    assert data["tts_characters"] == 42
    # error rate not tracked yet
    assert data["error_rate"] is None
    # we always include a numeric prev_* even if 0
    assert data["prev_total_calls"] >= 0


@pytest.mark.asyncio
async def test_summary_empty_window(db_client):
    r = await db_client.get("/api/v1/usage/summary?days=1", headers=_admin_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["total_calls"] == 0
    assert data["total_tokens"] == 0
    assert data["avg_latency_ms"] is None


@pytest.mark.asyncio
async def test_timeseries_groups_by_service(db_client, seeded):
    r = await db_client.get("/api/v1/usage/timeseries?days=7", headers=_admin_headers())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["days"] == 7
    # zero-fill: 7 days requested → 7 or 8 buckets (depending on 'now' boundary)
    assert len(data["points"]) >= 7
    seen = {svc for p in data["points"] for svc in p["by_service"]}
    assert "svc-a" in seen
    assert "svc-b" in seen
    assert set(data["top_services"]) >= {"svc-a", "svc-b"}


@pytest.mark.asyncio
async def test_top_keys_orders_by_calls(db_client, seeded):
    r = await db_client.get("/api/v1/usage/top-keys?days=7", headers=_admin_headers())
    assert r.status_code == 200, r.text
    data = r.json()
    rows = data["rows"]
    assert len(rows) == 2
    # mn-key has 2 calls vs legacy-key's 1
    assert rows[0]["label"] == "mn-key"
    assert rows[0]["mode"] == "m:n"
    assert rows[0]["calls"] == 2
    assert rows[1]["label"] == "legacy-key"
    assert rows[1]["mode"] == "legacy"


@pytest.mark.asyncio
async def test_top_keys_respects_limit(db_client, seeded):
    r = await db_client.get(
        "/api/v1/usage/top-keys?days=7&limit=1", headers=_admin_headers()
    )
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 1


@pytest.mark.asyncio
async def test_summary_rejects_oob_days(db_client):
    r = await db_client.get("/api/v1/usage/summary?days=0", headers=_admin_headers())
    assert r.status_code in (400, 422)
    r = await db_client.get("/api/v1/usage/summary?days=999", headers=_admin_headers())
    assert r.status_code in (400, 422)
