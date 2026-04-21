"""Admin /services/catalog — shape + aggregate math."""

from __future__ import annotations

import secrets as _secrets

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


@pytest.mark.asyncio
async def test_catalog_empty_when_no_instances_seeded_beyond_fixture(api_client, mock_vllm):
    """Fixture seeds one 'qwen3.5 test instance'. Catalog returns at least that row."""
    r = await api_client.get("/api/v1/services/catalog")
    assert r.status_code == 200, r.text
    data = r.json()
    names = [d["instance_name"] for d in data]
    assert "qwen3.5 test instance" in names


@pytest.mark.asyncio
async def test_catalog_aggregates_across_grants(api_client, mock_vllm):
    """Two keys with grants on one instance + one pack each → sums reflect both."""
    sf = api_client.app.state.async_session_factory

    # Fresh instance + two keys each with a grant + a pack.
    async with sf() as s:
        inst = ServiceInstance(
            source_type="model", source_name="aggx", name="aggx",
            type="llm", category="llm", meter_dim="tokens", status="active",
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)

        for prefix, used in [("sk-aa", 100), ("sk-bb", 200)]:
            raw = f"{prefix}-{_secrets.token_hex(4)}"
            key = InstanceApiKey(
                instance_id=None, label="k",
                key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
                key_prefix=raw[:10], is_active=True,
            )
            s.add(key)
            await s.commit()
            await s.refresh(key)
            g = ApiKeyGrant(api_key_id=key.id, instance_id=inst.id)
            s.add(g)
            await s.commit()
            await s.refresh(g)
            s.add(ResourcePack(
                grant_id=g.id, name=f"pack-{prefix}",
                total_units=1000, used_units=used,
            ))
            await s.commit()

    r = await api_client.get("/api/v1/services/catalog")
    row = next(d for d in r.json() if d["instance_name"] == "aggx")
    assert row["total_grants"] == 2
    assert row["active_grants"] == 2
    assert row["total_units"] == 2000
    assert row["used_units"] == 300
    assert row["remaining_units"] == 1700
