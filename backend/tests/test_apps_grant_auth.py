"""POST /v1/apps/{name} grant auth + quota gating."""

from __future__ import annotations

import bcrypt
import pytest
from unittest.mock import AsyncMock, patch

from src.models.api_gateway import ApiKeyGrant, ResourcePack
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


def _hash(token: str) -> str:
    return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()


@pytest.fixture
async def app_service(db_session):
    svc = ServiceInstance(
        source_type="workflow", source_name="x",
        name="echo-app", type="inference", status="active",
        category="app", meter_dim="calls",
        workflow_id=1,
        workflow_snapshot={"nodes": [], "edges": []},
        exposed_inputs=[],
    )
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)
    return svc


async def _make_key(db_session, prefix="sk-app1234"):
    raw = prefix + "abcdef"
    key = InstanceApiKey(
        instance_id=None, label="t", key_hash=_hash(raw),
        key_prefix=raw[:10], is_active=True,
    )
    db_session.add(key)
    await db_session.commit()
    await db_session.refresh(key)
    return raw, key


@pytest.mark.asyncio
async def test_no_grant_returns_404(db_client, db_session, app_service):
    raw, _ = await _make_key(db_session, prefix="sk-no123456")
    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"out": "ok"}),
    ):
        r = await db_client.post(
            f"/v1/apps/{app_service.name}",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
    # No grant linking this key to this service → resolver fails with 404.
    assert r.status_code in (403, 404), r.text


@pytest.mark.asyncio
async def test_paused_grant_blocks_call(
    db_client, db_session, app_service,
):
    raw, key = await _make_key(db_session, prefix="sk-pa345678")
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, service_id=app_service.id, status="paused",
    ))
    await db_session.commit()

    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"out": "ok"}),
    ):
        r = await db_client.post(
            f"/v1/apps/{app_service.name}",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
    # Paused grants are filtered by the resolver, so this looks like "no grant".
    assert r.status_code in (403, 404)


@pytest.mark.asyncio
async def test_active_grant_runs_and_charges_one_call(
    db_client, db_session, app_service,
):
    raw, key = await _make_key(db_session, prefix="sk-ok456789")
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, service_id=app_service.id, status="active",
    ))
    await db_session.flush()
    grant = (await db_session.execute(
        ApiKeyGrant.__table__.select().where(
            ApiKeyGrant.api_key_id == key.id
        )
    )).first()
    pack = ResourcePack(
        grant_id=grant.id, name="100 calls",
        total_units=100, used_units=0, source="purchased",
    )
    db_session.add(pack)
    await db_session.commit()

    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"out": "ok"}),
    ):
        r = await db_client.post(
            f"/v1/apps/{app_service.name}",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"out": "ok"}

    # 1 call should have been consumed (units=1, dim='calls').
    await db_session.refresh(pack)
    assert pack.used_units == 1


@pytest.mark.asyncio
async def test_retired_service_returns_410(
    db_client, db_session, app_service,
):
    """v3 lifecycle: retired services 410, paused 403, deprecated still serves."""
    raw, key = await _make_key(db_session, prefix="sk-rt567890")
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, service_id=app_service.id, status="active",
    ))
    app_service.status = "retired"
    await db_session.commit()

    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"out": "ok"}),
    ):
        r = await db_client.post(
            f"/v1/apps/{app_service.name}",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
    assert r.status_code == 410
