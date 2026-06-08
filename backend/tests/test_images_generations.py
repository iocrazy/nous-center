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


def test_extract_image_urls_prefers_exposed_outputs():
    """exposed_outputs 指定产图终端 → 精确取它,不受其他节点 image_url 干扰。"""
    result = {
        "outputs": {
            "img": {"image_url": "/files/images/INPUT.png"},   # image_input 上传图回显
            "dec": {"image_url": "/files/images/OUT.png"},      # 真产图
        }
    }
    snapshot = {"nodes": [
        {"id": "img", "type": "image_input", "data": {}},
        {"id": "dec", "type": "flux2_vae_decode", "data": {}},
    ]}
    exposed_outputs = [{"node_id": "dec", "input_name": "image_url"}]
    assert _extract_image_urls(result, snapshot, exposed_outputs) == ["/files/images/OUT.png"]


def test_extract_image_urls_skips_image_input_echo():
    """无 exposed_outputs 兜底:跳过 image_input 节点的上传图回显(#372 外部路径版)。"""
    result = {
        "outputs": {
            "img": {"image_url": "/files/images/INPUT.png"},   # 必须被跳过
            "up": {"image_url": "/files/images/UPSCALED.png"},  # seedvr2_upscale 产图
        }
    }
    snapshot = {"nodes": [
        {"id": "img", "type": "image_input", "data": {}},
        {"id": "up", "type": "seedvr2_upscale", "data": {}},
    ]}
    assert _extract_image_urls(result, snapshot, None) == ["/files/images/UPSCALED.png"]


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


# --- 图输入 / 无 prompt 服务(P4:编辑/增强/角度) ---


@pytest.fixture
async def edit_service(db_session):
    """图片编辑式服务:image(image_input)+ prompt(encode)→ dec 产图。"""
    svc = ServiceInstance(
        source_type="workflow", source_name="edit",
        name="edit-svc", type="inference", status="active",
        category="image", meter_dim="images", workflow_id=2,
        workflow_snapshot={
            "nodes": [
                {"id": "img", "type": "image_input", "data": {}},
                {"id": "enc", "type": "flux2_encode_prompt", "data": {}},
                {"id": "dec", "type": "flux2_vae_decode", "data": {}},
            ],
            "edges": [],
        },
        exposed_inputs=[
            {"node_id": "img", "key": "image", "input_name": "image", "type": "image"},
            {"node_id": "enc", "key": "prompt", "input_name": "text", "type": "string"},
        ],
        exposed_outputs=[
            {"node_id": "dec", "key": "image_url", "input_name": "image_url", "type": "string"}
        ],
    )
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)
    return svc


@pytest.fixture
async def upscale_service(db_session):
    """SeedVR2 细节增强式服务:image + resolution,无 prompt → up 产图。"""
    svc = ServiceInstance(
        source_type="workflow", source_name="up",
        name="upscale-svc", type="inference", status="active",
        category="image", meter_dim="images", workflow_id=3,
        workflow_snapshot={
            "nodes": [
                {"id": "img", "type": "image_input", "data": {}},
                {"id": "up", "type": "seedvr2_upscale", "data": {}},
            ],
            "edges": [],
        },
        exposed_inputs=[
            {"node_id": "img", "key": "image", "input_name": "image", "type": "image"},
            {"node_id": "up", "key": "resolution", "input_name": "resolution", "type": "int"},
        ],
        exposed_outputs=[
            {"node_id": "up", "key": "image_url", "input_name": "image_url", "type": "string"}
        ],
    )
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)
    return svc


@pytest.mark.asyncio
async def test_images_injects_image_and_prompt(db_client, db_session, edit_service):
    """编辑服务:image + prompt 都注入对应节点(image_input echo 不当输出返回)。"""
    raw, key = await _make_key(db_session, "sk-imgedit01")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=edit_service.id, status="active"))
    await db_session.commit()
    captured = {}

    async def _fake_execute(self):
        captured["nodes"] = self.nodes
        return {"outputs": {
            "img": {"image_url": "/files/images/INPUT.png"},   # 回显输入图,必须被跳过
            "dec": {"image_url": "/files/images/EDITED.png"},  # 真产图
        }}

    with patch("src.services.workflow_executor.WorkflowExecutor.execute", new=_fake_execute):
        r = await db_client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {raw}"},
            json={
                "model": edit_service.name,
                "prompt": "make it snow",
                "image": "data:image/png;base64,iVBORw0KGgo=",
            },
        )
    assert r.status_code == 200, r.text
    # exposed_outputs=dec → 取真产图,不是输入图回显。
    assert r.json()["data"][0]["url"].endswith("/files/images/EDITED.png")
    # image + prompt 都注入了对应节点 data。
    by_id = {n["id"]: n for n in captured["nodes"]}
    assert by_id["img"]["data"]["image"].startswith("data:image/png;base64,")
    assert by_id["enc"]["data"]["text"] == "make it snow"


@pytest.mark.asyncio
async def test_images_no_prompt_service_works(db_client, db_session, upscale_service):
    """超分服务无 prompt:image + resolution(extra 字段)注入,不触发 no_prompt 报错。"""
    raw, key = await _make_key(db_session, "sk-imgup0123")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=upscale_service.id, status="active"))
    await db_session.commit()
    captured = {}

    async def _fake_execute(self):
        captured["nodes"] = self.nodes
        return {"outputs": {
            "img": {"image_url": "/files/images/INPUT.png"},
            "up": {"image_url": "/files/images/UP.png"},
        }}

    with patch("src.services.workflow_executor.WorkflowExecutor.execute", new=_fake_execute):
        r = await db_client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {raw}"},
            json={
                "model": upscale_service.name,
                "image": "data:image/png;base64,iVBORw0KGgo=",
                "resolution": 1440,
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["url"].endswith("/files/images/UP.png")
    by_id = {n["id"]: n for n in captured["nodes"]}
    assert by_id["img"]["data"]["image"].startswith("data:image/png;base64,")
    assert by_id["up"]["data"]["resolution"] == 1440
