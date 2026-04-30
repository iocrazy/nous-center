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


@pytest.mark.asyncio
async def test_admin_session_bypasses_bearer_for_playground(
    db_client, db_session, app_service,
):
    """Playground tab uses session cookie, not Bearer — admin must reach the
    executor without an API key, without grant lookup, and without charging
    quota. Pre-fix the request died at FastAPI header validation with
    'Field required' because Authorization was Header(...)."""
    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"out": "playground"}),
    ):
        r = await db_client.post(
            f"/v1/apps/{app_service.name}",
            json={},  # no Authorization header — relies on admin cookie
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"out": "playground"}


@pytest.mark.asyncio
async def test_no_auth_at_all_returns_401(db_client, monkeypatch):
    """Without admin cookie AND without Bearer the request must be rejected.
    Tests force ADMIN_PASSWORD='' which makes admin auth permissive; flip
    that for this case so the unauth path is actually exercised."""
    monkeypatch.setenv("ADMIN_PASSWORD", "real-pw")
    from src.config import get_settings
    get_settings.cache_clear()
    try:
        r = await db_client.post("/v1/apps/anything", json={})
    finally:
        get_settings.cache_clear()
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_input_merge_writes_to_node_primary_slot(
    db_client, db_session,
):
    """The publish dialog used to hard-code `input_name='value'` for every
    exposed param. text_input reads `data.text`, so the merge silently
    missed and the node returned its frozen value — making the LLM answer
    the OLD prompt baked into publish instead of the caller's new input.
    apps.py now also writes to the node's primary slot when `slot` doesn't
    match, so legacy services keep working without re-publish."""
    svc = ServiceInstance(
        source_type="workflow", source_name="x",
        name="slot-svc", type="inference", status="active",
        category="app", meter_dim="calls",
        workflow_id=2,
        workflow_snapshot={
            "nodes": [
                {"id": "in1", "type": "text_input", "data": {"text": "frozen-old-prompt"}},
                {"id": "out1", "type": "text_output", "data": {}},
            ],
            "edges": [{"source": "in1", "target": "out1", "sourceHandle": "text", "targetHandle": "text"}],
        },
        exposed_inputs=[
            # Legacy schema: input_name='value' but text_input reads 'text'
            {"node_id": "in1", "key": "prompt", "input_name": "value",
             "type": "string", "required": True},
        ],
    )
    db_session.add(svc)
    raw, key = await _make_key(db_session, prefix="sk-slot12345")
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, service_id=svc.id, status="active",
    ))
    await db_session.commit()

    r = await db_client.post(
        f"/v1/apps/{svc.name}",
        headers={"Authorization": f"Bearer {raw}"},
        json={"prompt": "fresh-caller-input"},
    )
    assert r.status_code == 200, r.text
    # text_output emits {"text": inputs["text"]} — should be the caller's
    # new input, not the frozen "frozen-old-prompt".
    payload = r.json()
    out_bucket = payload.get("outputs", {}).get("out1", {})
    assert out_bucket.get("text") == "fresh-caller-input"
