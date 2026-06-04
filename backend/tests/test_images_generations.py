"""POST /v1/images/generations — OpenAI/火山兼容图像端点(M:N key scope + 出图)。

端点逻辑走共享核心 run_published_workflow(与 /v1/apps 同),这里只测端点自身:
key scope 解析、prompt→exposed input、image_url→OpenAI {data:[{url}]} 转换。
WorkflowExecutor.execute 被 mock(真出图靠 standalone 真机,CI 无 GPU)。
"""

from __future__ import annotations

import bcrypt
import pytest
from unittest.mock import AsyncMock, patch

from src.api.routes.openai_compat import _extract_image_urls, _pick_prompt_input_key
from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance


def _hash(token: str) -> str:
    return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()


# --- 纯函数 ---


def test_pick_prompt_input_key_prefers_string_type():
    inputs = [{"key": "seed", "type": "int"}, {"key": "prompt", "type": "string"}]
    assert _pick_prompt_input_key(inputs) == "prompt"


def test_pick_prompt_input_key_falls_back_to_first():
    assert _pick_prompt_input_key([{"key": "x", "type": "whatever"}]) == "x"
    assert _pick_prompt_input_key([]) is None
    assert _pick_prompt_input_key(None) is None


def test_extract_image_urls_from_outputs():
    result = {
        "outputs": {
            "in": {"text": "cat"},
            "dec": {"image_url": "/files/images/a.png", "width": 1024},
            "out": {"image_url": "/files/images/b.png"},
        }
    }
    assert _extract_image_urls(result) == ["/files/images/a.png", "/files/images/b.png"]
    assert _extract_image_urls({}) == []
    assert _extract_image_urls({"outputs": {}}) == []


# --- 端点集成 ---


@pytest.fixture
async def image_service(db_session):
    svc = ServiceInstance(
        source_type="workflow", source_name="x",
        name="img-svc", type="inference", status="active",
        category="app", meter_dim="calls",
        workflow_id=1,
        workflow_snapshot={
            "nodes": [
                {"id": "in1", "type": "text_input", "data": {}},
                {"id": "dec", "type": "flux2_vae_decode", "data": {}},
            ],
            "edges": [],
        },
        exposed_inputs=[
            {"node_id": "in1", "key": "prompt", "input_name": "text", "type": "string"}
        ],
        exposed_outputs=[
            {"node_id": "dec", "key": "image_url", "input_name": "image_url", "type": "string"}
        ],
    )
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)
    return svc


async def _make_key(db_session, prefix):
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
async def test_images_no_grant_returns_404(db_client, db_session, image_service):
    raw, _ = await _make_key(db_session, "sk-imgno123")
    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"outputs": {"dec": {"image_url": "/files/x.png"}}}),
    ):
        r = await db_client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {raw}"},
            json={"model": image_service.name, "prompt": "a cat"},
        )
    # 无 grant 把 key 连到该 service → 解析 404(火山式 scope:key 只能访问被授权的 model)。
    assert r.status_code in (403, 404), r.text


@pytest.mark.asyncio
async def test_images_active_grant_returns_openai_shape(db_client, db_session, image_service):
    raw, key = await _make_key(db_session, "sk-imgok123")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=image_service.id, status="active"))
    await db_session.commit()
    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"outputs": {
            "in1": {"text": "a cat"},
            "dec": {"image_url": "/files/images/2026-06-04/abc.png"},
        }}),
    ):
        r = await db_client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {raw}"},
            json={"model": image_service.name, "prompt": "a cat"},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "created" in data
    assert isinstance(data["data"], list) and len(data["data"]) == 1
    assert data["data"][0]["url"].endswith("/files/images/2026-06-04/abc.png")


@pytest.mark.asyncio
async def test_images_no_image_output_errors(db_client, db_session, image_service):
    raw, key = await _make_key(db_session, "sk-imgno567")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=image_service.id, status="active"))
    await db_session.commit()
    with patch(
        "src.services.workflow_executor.WorkflowExecutor.execute",
        new=AsyncMock(return_value={"outputs": {"in1": {"text": "a cat"}}}),  # 无 image_url
    ):
        r = await db_client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {raw}"},
            json={"model": image_service.name, "prompt": "a cat"},
        )
    assert r.status_code >= 400, r.text
