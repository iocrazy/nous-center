"""Lane B-T3 · /v1/chat/completions dispatch & resolver wiring.

Covers the M:N path (new): keys with no legacy binding are resolved via
ApiKeyGrant by matching request.model against ServiceInstance.name.
Dispatch by source_type: model is served; workflow/app return 501 until
their handlers land.

Legacy 1:1 bound keys are tested implicitly by test_responses_agent_binding
and other suites — that path runs unchanged through the new dep.
"""

from __future__ import annotations

import secrets as _secrets

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


async def _make_mn_key(session_factory, *, instance_name: str, source_type: str = "model"):
    """Create an M:N key with an active grant on a fresh instance. Returns
    (raw_key, instance_id)."""
    raw_key = f"sk-mn-{_secrets.token_hex(8)}"
    async with session_factory() as s:
        inst = ServiceInstance(
            source_type=source_type,
            source_name=instance_name if source_type == "model" else None,
            source_id=1 if source_type != "model" else None,
            name=instance_name,
            type="llm",
            status="active",
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)

        key = InstanceApiKey(
            instance_id=None,  # M:N
            label="mn",
            key_hash=bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw_key[:10],
            is_active=True,
        )
        s.add(key)
        await s.commit()
        await s.refresh(key)

        grant = ApiKeyGrant(
            api_key_id=key.id, service_id=inst.id, status="active",
        )
        s.add(grant)
        await s.commit()

        return raw_key, inst.id


@pytest.mark.asyncio
async def test_mn_key_unknown_model_returns_404(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    raw_key, _ = await _make_mn_key(sf, instance_name="qwen-preview")

    resp = await api_client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_mn_key_workflow_instance_returns_501(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    raw_key, _ = await _make_mn_key(
        sf, instance_name="my-workflow", source_type="workflow",
    )
    resp = await api_client.post(
        "/v1/chat/completions",
        json={"model": "my-workflow", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 501, resp.text
    assert "workflow" in resp.text.lower()


@pytest.mark.asyncio
async def test_mn_key_app_instance_returns_501(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    raw_key, _ = await _make_mn_key(
        sf, instance_name="my-app", source_type="app",
    )
    resp = await api_client.post(
        "/v1/chat/completions",
        json={"model": "my-app", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 501, resp.text
    assert "app" in resp.text.lower()


@pytest.mark.asyncio
async def test_mn_key_missing_model_returns_404(api_client, mock_vllm):
    sf = api_client.app.state.async_session_factory
    raw_key, _ = await _make_mn_key(sf, instance_name="qwen-preview")

    resp = await api_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp.status_code == 404, resp.text
