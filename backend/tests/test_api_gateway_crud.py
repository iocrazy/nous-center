"""Lane E backend · admin CRUD for grants/packs/alerts + /services/me.

Covers the happy paths the UI needs. Admin token is empty in tests (dev
mode), so we don't exercise the auth boundary here — deps_admin.py is
tested separately.
"""

from __future__ import annotations

import secrets as _secrets

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


async def _seed_instance_and_key(sf, *, name: str, mn: bool = True):
    raw = f"sk-t-{_secrets.token_hex(6)}"
    async with sf() as s:
        inst = ServiceInstance(
            source_type="model", source_name=name, name=name,
            type="llm", category="llm", meter_dim="tokens", status="active",
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)
        key = InstanceApiKey(
            instance_id=None if mn else inst.id, label="t",
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10], is_active=True,
        )
        s.add(key)
        await s.commit()
        await s.refresh(key)
        return raw, key.id, inst.id


@pytest.mark.asyncio
async def test_create_and_list_grant(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    _, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5")

    r = await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )
    assert r.status_code == 201, r.text
    g = r.json()
    assert g["instance_name"] == "gpt5"
    assert g["status"] == "active"

    r2 = await api_client.get(f"/api/v1/keys/{key_id}/grants")
    assert r2.status_code == 200
    assert len(r2.json()) == 1


@pytest.mark.asyncio
async def test_duplicate_grant_returns_409(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    _, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5")

    r1 = await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )
    assert r1.status_code == 201
    r2 = await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_patch_grant_pauses_then_resumes(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    _, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5")
    r = await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )
    gid = r.json()["id"]

    r = await api_client.patch(f"/api/v1/grants/{gid}", json={"status": "paused"})
    assert r.status_code == 200
    assert r.json()["status"] == "paused"
    assert r.json()["paused_at"] is not None

    r = await api_client.patch(f"/api/v1/grants/{gid}", json={"status": "active"})
    assert r.json()["status"] == "active"
    assert r.json()["paused_at"] is None


@pytest.mark.asyncio
async def test_pack_crud_and_remaining(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    _, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5")
    g = (await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )).json()

    r = await api_client.post(
        f"/api/v1/grants/{g['id']}/packs",
        json={"name": "free trial", "total_units": 1000, "source": "free_trial"},
    )
    assert r.status_code == 201, r.text
    pack = r.json()
    assert pack["remaining_units"] == 1000

    # Deduct some used_units directly in DB to verify remaining math.
    async with sf() as s:
        p = await s.get(ResourcePack, pack["id"])
        p.used_units = 200
        await s.commit()

    packs = (await api_client.get(f"/api/v1/grants/{g['id']}/packs")).json()
    assert packs[0]["remaining_units"] == 800


@pytest.mark.asyncio
async def test_alert_crud(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    _, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5")
    g = (await api_client.post(
        f"/api/v1/keys/{key_id}/grants", json={"instance_id": inst_id},
    )).json()

    r = await api_client.post(
        f"/api/v1/grants/{g['id']}/alerts", json={"threshold_percent": 80},
    )
    assert r.status_code == 201, r.text
    rule = r.json()
    assert rule["enabled"] is True

    # Disable it via PATCH.
    r = await api_client.patch(
        f"/api/v1/alerts/{rule['id']}", json={"enabled": False},
    )
    assert r.json()["enabled"] is False

    # Invalid threshold rejected (400 or 422 depending on validation layer).
    r = await api_client.post(
        f"/api/v1/grants/{g['id']}/alerts", json={"threshold_percent": 150},
    )
    assert r.status_code in (400, 422)


@pytest.mark.asyncio
async def test_services_me_mn_key(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    raw, key_id, inst_id = await _seed_instance_and_key(sf, name="gpt5", mn=True)
    # Seed a grant + pack directly so we don't go through admin endpoints.
    async with sf() as s:
        grant = ApiKeyGrant(api_key_id=key_id, instance_id=inst_id)
        s.add(grant)
        await s.commit()
        await s.refresh(grant)
        s.add(ResourcePack(
            grant_id=grant.id, name="p1", total_units=500, used_units=100,
        ))
        await s.commit()

    r = await api_client.get(
        "/api/v1/services/me",
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 200, r.text
    services = r.json()
    assert len(services) == 1
    svc = services[0]
    assert svc["instance_name"] == "gpt5"
    assert svc["grant_status"] == "active"
    assert svc["total_units"] == 500
    assert svc["used_units"] == 100
    assert svc["remaining_units"] == 400


@pytest.mark.asyncio
async def test_services_me_legacy_key(api_client, bearer_headers, mock_vllm):
    """Legacy 1:1 key returns a single synthetic row with status=legacy."""
    r = await api_client.get("/api/v1/services/me", headers=bearer_headers)
    assert r.status_code == 200
    assert r.json()[0]["grant_status"] == "legacy"
