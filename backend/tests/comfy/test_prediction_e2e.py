"""Task 7: predictions 端到端 + 取消打通。

comfy_template 服务(Task 3)+ 桥节点(Task 6)经统一 `POST /v1/services/{name}/predictions`
（服务层 API spec PR-2,见 `src/api/routes/predictions.py`）真正跑通:respond-async → 202 →
轮询 `/v1/predictions/{id}` → succeeded + 视频 url。cancel 打通 `POST /v1/predictions/{id}/cancel`
→ comfy_template 服务 running 态时额外转发 `ComfyClient.interrupt()`(Task 7 本文件新增行为,
见 `predictions.py:cancel_prediction`)。

鉴权:`/v1/*` 走 `verify_bearer_token_any`,legacy 1:1 key 已删(见
`tests/test_prediction_service_pr2.py`),只认 M:N `InstanceApiKey` + `ApiKeyGrant`——本文件
的 `_make_service_and_key` 照抄该文件 `prediction_service` fixture 的建 key/建 grant 方式。
DB 用 `client` fixture 背后进程内 memoized 的共享 session factory(同
`test_bridge_node.py::test_load_template_real_db_lookup`),不是自建临时 engine。
"""
from __future__ import annotations

import asyncio
import secrets as _secrets

import bcrypt
import pytest
from sqlalchemy import select

import src.api.routes.comfy_templates as comfy_templates_route
import src.services.nodes.comfy_bridge as nb
from src.models.api_gateway import ApiKeyGrant
from src.models.database import get_session_factory
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from tests.comfy.test_bridge_node import FakeClient

WF = {
    "138": {"class_type": "T", "inputs": {"value": ""}},
    "92": {"class_type": "SaveVideo", "inputs": {}},
}
MAPPING = {"exposed_params": [
    {"key": "prompt", "label": "提示词", "type": "string",
     "comfy_node_id": "138", "comfy_input": "value", "required": True},
]}


async def _make_service_and_key(client, name: str) -> str:
    """建一个 comfy_template 服务(经 API,含 mapping)+ 一把 M:N bearer key 授权它。返回原始 key。"""
    r = await client.post("/api/v1/comfy-templates", json={"name": name, "workflow": WF})
    assert r.status_code == 201, r.text
    tid = int(r.json()["id"])
    r2 = await client.put(f"/api/v1/comfy-templates/{tid}/mapping", json=MAPPING)
    assert r2.status_code == 200, r2.text

    raw = f"sk-comfy-{_secrets.token_hex(8)}"
    session_factory = get_session_factory()
    async with session_factory() as s:
        svc = (await s.execute(
            select(ServiceInstance).where(ServiceInstance.name == name))).scalar_one()
        key = InstanceApiKey(
            instance_id=None, label="t",  # M:N — 走 grant(legacy 1:1 已删)
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10], is_active=True)
        s.add(key)
        await s.commit()
        await s.refresh(key)
        s.add(ApiKeyGrant(api_key_id=key.id, service_id=svc.id, status="active"))
        await s.commit()
    return raw


async def _poll_prediction(client, pid, headers, timeout: float = 5.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    body: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/v1/predictions/{pid}", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("succeeded", "failed", "canceled"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"prediction {pid} 未在 {timeout}s 内到终态,last={body}")


def _patch_no_thumbnail(monkeypatch) -> None:
    """跳过真 ffmpeg——FakeClient.download() 吐的不是合法 mp4 字节,extract_first_frame
    会静默 fail(设计如此,不影响本文件断言),但真去 spawn 一个进程没必要拖慢测试。"""
    async def _no_thumbnail(_path):
        return None
    monkeypatch.setattr(nb, "extract_first_frame", _no_thumbnail)


@pytest.mark.asyncio
async def test_async_prediction_completes_with_video(client, monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(nb, "get_client", lambda: fc)
    _patch_no_thumbnail(monkeypatch)

    raw_key = await _make_service_and_key(client, "e2e-video")
    headers = {"Authorization": f"Bearer {raw_key}"}

    r = await client.post(
        "/v1/services/e2e-video/predictions", json={"input": {"prompt": "hi"}},
        headers={**headers, "Prefer": "respond-async"})
    assert r.status_code == 202, r.text
    pid = r.json()["id"]

    final = await _poll_prediction(client, pid, headers)
    assert final["status"] == "succeeded", final
    assert "mp4" in str(final["output"])


@pytest.mark.asyncio
async def test_predictions_sync_mode_also_returns_video(client, monkeypatch):
    """无 Prefer(同步阻塞)路径同样跑通——回归 predictions.py 的两条模式共享同一执行链路。"""
    fc = FakeClient()
    monkeypatch.setattr(nb, "get_client", lambda: fc)
    _patch_no_thumbnail(monkeypatch)

    raw_key = await _make_service_and_key(client, "e2e-video-sync")
    headers = {"Authorization": f"Bearer {raw_key}"}

    r = await client.post(
        "/v1/services/e2e-video-sync/predictions", json={"input": {"prompt": "hi"}},
        headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "succeeded", body
    assert "mp4" in str(body["output"])


class _InterruptFakeClient(FakeClient):
    """FakeClient 子类:wait() 卡在测试控制的 asyncio.Event 上(不用 sleep 赌时序),
    interrupt() 计数供 cancel 转发断言。"""

    def __init__(self, gate: asyncio.Event):
        super().__init__()
        self._gate = gate
        self.interrupt_calls = 0

    async def wait(self, prompt_id, *, timeout_s, interval_s=2.0):
        await self._gate.wait()
        return await super().wait(prompt_id, timeout_s=timeout_s, interval_s=interval_s)

    async def interrupt(self) -> None:
        self.interrupt_calls += 1


@pytest.mark.asyncio
async def test_cancel_forwards_interrupt_and_reaches_cancelled(client, monkeypatch):
    """取消一个卡在 ComfyClient.wait() 上的 running prediction:
    - cancel 端点(predictions.py cancel_prediction)在发现 task 归属服务是 comfy_template
      且 status=running 时,额外 await get_client().interrupt()(get_client 从
      comfy_templates 路由模块 import——同一个被 monkeypatch 的 FakeClient 实例)。
    - 放行卡住的 wait() 后,workflow_runner 的 cancel-race guard 必须 honor 已经落的
      cancelled,而不是被"成功跑完"的路径覆盖回 succeeded。
    """
    gate = asyncio.Event()
    fc = _InterruptFakeClient(gate)
    monkeypatch.setattr(nb, "get_client", lambda: fc)
    monkeypatch.setattr(comfy_templates_route, "get_client", lambda: fc)
    _patch_no_thumbnail(monkeypatch)

    raw_key = await _make_service_and_key(client, "e2e-cancel")
    headers = {"Authorization": f"Bearer {raw_key}"}

    r = await client.post(
        "/v1/services/e2e-cancel/predictions", json={"input": {"prompt": "hi"}},
        headers={**headers, "Prefer": "respond-async"})
    assert r.status_code == 202, r.text
    pid = r.json()["id"]

    # 等后台任务真正进 running(= 已提交 ComfyUI,卡在 gate 上)——interrupt 转发只在
    # running 态触发,过早 cancel(还 queued)测的就不是这条分支了。
    deadline = asyncio.get_event_loop().time() + 5.0
    status = None
    while asyncio.get_event_loop().time() < deadline:
        status = (await client.get(f"/v1/predictions/{pid}", headers=headers)).json()["status"]
        if status == "processing":
            break
        await asyncio.sleep(0.02)
    assert status == "processing", f"prediction 未在超时前进入 running(last={status})"

    cancel_resp = await client.post(f"/v1/predictions/{pid}/cancel", headers=headers)
    assert cancel_resp.status_code == 200, cancel_resp.text
    assert cancel_resp.json()["status"] == "canceled"
    assert fc.interrupt_calls == 1

    # 放行卡住的 wait()——run_workflow_task 的 success 路径会 refresh 看到 cancelled 并 honor。
    gate.set()
    final = await _poll_prediction(client, pid, headers)
    assert final["status"] == "canceled", final
    assert final["output"] is None  # 未终态成功,不该有 output


@pytest.mark.asyncio
async def test_cancel_on_non_comfy_service_skips_interrupt(client, monkeypatch):
    """非 comfy_template 服务(普通 workflow 服务)cancel 时不该去碰 comfy get_client——
    回归守护:cancel_prediction 的新分支必须按 ServiceInstance.source_type 精确判断,
    不是对所有 running task 一律尝试 interrupt。"""
    from src.models.execution_task import ExecutionTask

    session_factory = get_session_factory()
    async with session_factory() as s:
        inst = ServiceInstance(
            source_type="workflow", source_name="plain-wf", name="plain-svc",
            type="workflow", status="active",
            workflow_snapshot={"nodes": {}, "edges": []},
            exposed_inputs=[], exposed_outputs=[],
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)
        raw = f"sk-plain-{_secrets.token_hex(8)}"
        key = InstanceApiKey(
            instance_id=None, label="t",
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10], is_active=True)
        s.add(key)
        await s.commit()
        await s.refresh(key)
        s.add(ApiKeyGrant(api_key_id=key.id, service_id=inst.id, status="active"))
        await s.commit()

        task = ExecutionTask(
            workflow_id=inst.source_id, api_key_id=key.id, workflow_name=inst.name,
            status="running", nodes_total=1,
        )
        s.add(task)
        await s.commit()
        await s.refresh(task)
        task_id = task.id

    called = {"n": 0}

    def _boom():
        called["n"] += 1
        raise AssertionError("comfy_templates.get_client 不该在非 comfy_template 服务上被调用")
    monkeypatch.setattr(comfy_templates_route, "get_client", _boom)

    resp = await client.post(f"/v1/predictions/{task_id}/cancel",
                             headers={"Authorization": f"Bearer {raw}"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "canceled"
    assert called["n"] == 0


def test_task_to_dict_surfaces_video_thumbnails_and_url():
    """Task 7 Step 3:comfy 桥节点(comfy_bridge.py ComfyUIWorkflowNode)的产物形状是
    `thumbnails`(ffmpeg 首帧)/`video_url`(原片)/`items[].url`(全量),不是老的单一
    `image_url`。execution_task_serialize._image_urls 补齐这三种 key 前,
    这些 url 不进 output_thumbnails——不只是显示 bug:它们也不会被
    collect_referenced_image_uuids(同一 _image_urls)识别为"仍被引用",24h orphan
    reaper(main.py image_orphan_reap_loop)会把它们当无引用误删,即便任务历史仍在引用。
    """
    from datetime import datetime, timezone

    from src.api.routes.execution_tasks import _task_to_dict
    from src.models.execution_task import ExecutionTask

    now = datetime.now(timezone.utc)
    t = ExecutionTask(
        id=41, workflow_id=1, workflow_name="video-svc", status="completed",
        nodes_total=2, nodes_done=2, current_node=None,
        result={"outputs": {"out": {
            "items": [{"url": "/files/out.mp4?t=1", "kind": "video",
                       "filename": "out.mp4", "node_id": "92", "class_type": "SaveVideo"}],
            "video_url": "/files/out.mp4?t=1",
            "thumbnails": ["/files/thumb.png?t=1"],
            "seed": None,
        }}},
        error=None, duration_ms=100, created_at=now, updated_at=now,
    )
    d = _task_to_dict(t)
    assert "/files/thumb.png?t=1" in d["output_thumbnails"]
    assert "/files/out.mp4?t=1" in d["output_thumbnails"]
    # dedup:items[].url 和 video_url 是同一张(同一支视频原片),不该在 8 张缩略帽里占两格。
    assert d["output_thumbnails"].count("/files/out.mp4?t=1") == 1
