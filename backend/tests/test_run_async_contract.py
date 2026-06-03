"""Lane S: /run 纯异步契约（D17）回归。

回归风险：/run 从同步阻塞改成 202 + task_id。本测试守住
enqueue → poll /tasks/{id} → result 的端到端链路不断。
"""
import asyncio
import secrets as _secrets

import bcrypt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
async def async_client_with_db(tmp_path, monkeypatch):
    """Async test client with a real SQLite DB + workflow_runner pointed at it.

    workflow_runner opens its own session via create_session_factory(); we
    monkeypatch the symbol on src.services.workflow_runner so background runs
    write to the same SQLite DB the request handler reads from.
    """
    from src.api.main import create_app
    from src.models.database import Base, get_async_session

    db_path = tmp_path / "run_async.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with session_factory() as session:
            yield session

    test_app = create_app()
    from unittest.mock import MagicMock
    test_app.state.model_manager = MagicMock()
    test_app.dependency_overrides[get_async_session] = override_session

    # Pin workflow_runner's session factory to the test DB（它 import 时已绑引用）。
    from src.services import workflow_runner as _runner
    monkeypatch.setattr(_runner, "get_session_factory",
                        lambda: session_factory)
    # predictions SSE 流懒 import `src.models.database.get_session_factory`(每轮新 session)——
    # 补 patch 源,否则 SSE 走真 Postgres(测试用 SQLite)。
    import src.models.database as _db
    monkeypatch.setattr(_db, "get_session_factory", lambda: session_factory)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.session_factory = session_factory  # type: ignore[attr-defined]
        yield c

    await engine.dispose()


async def _poll_until_done(client, task_id, timeout=5.0):
    """轮询 /api/v1/tasks/{id} 直到 status 进入终态。"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} 未在 {timeout}s 内完成")


@pytest.mark.asyncio
async def test_execute_workflow_direct_returns_202_task_id(async_client_with_db):
    """POST /api/v1/workflows/execute → 202 + task_id（不再同步阻塞返回 result）。"""
    client = async_client_with_db
    resp = await client.post("/api/v1/workflows/execute", json={
        "name": "test-async",
        "nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hi"},
                   "position": {"x": 0, "y": 0}}],
        "edges": [],
    })
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "task_id" in body


@pytest.mark.asyncio
async def test_execute_workflow_direct_enqueue_poll_result(async_client_with_db):
    """端到端：enqueue → poll → 拿到 completed + result。"""
    client = async_client_with_db
    resp = await client.post("/api/v1/workflows/execute", json={
        "name": "e2e",
        "nodes": [{"id": "t1", "type": "text_input", "data": {"text": "hello"},
                   "position": {"x": 0, "y": 0}}],
        "edges": [],
    })
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    final = await _poll_until_done(client, task_id)
    assert final["status"] == "completed"
    assert final["result"] is not None
    assert final["nodes_done"] == 1


@pytest.fixture
async def prediction_service(async_client_with_db):
    """workflow 服务(图在 workflow_snapshot,dict-of-nodes 形)+ M:N key + grant + 一个 exposed_input。
    服务层 API spec PR-2:测统一 POST /services/{name}/predictions(取代旧 /run)。
    legacy rip:key 由 1:1 改 M:N(instance_id=None + ApiKeyGrant),legacy 分支已删。"""
    from src.models.api_gateway import ApiKeyGrant
    from src.models.instance_api_key import InstanceApiKey
    from src.models.service_instance import ServiceInstance

    session_factory = async_client_with_db.session_factory
    raw = f"sk-pred-{_secrets.token_hex(8)}"
    async with session_factory() as s:
        inst = ServiceInstance(
            source_type="workflow", source_name="pred-wf", name="pred-svc",
            type="workflow", status="active",
            workflow_snapshot={
                "nodes": {"t1": {"class_type": "text_input", "inputs": {"text": "冻结默认"}}},
                "edges": [],
            },
            exposed_inputs=[{"node_id": "t1", "key": "text", "input_name": "text",
                             "type": "string", "required": False}],
            exposed_outputs=[],
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)
        key = InstanceApiKey(
            instance_id=None, label="t",  # M:N — 走 grant
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10], is_active=True)
        s.add(key)
        await s.commit()
        await s.refresh(key)
        s.add(ApiKeyGrant(api_key_id=key.id, service_id=inst.id, status="active"))
        await s.commit()
    w = type("_W", (), {})()
    w.name = inst.name
    w.raw_key = raw
    return w


@pytest.mark.asyncio
async def test_predictions_async_returns_202(async_client_with_db, prediction_service):
    """Prefer: respond-async → 202 + prediction{id, status: processing/starting}。"""
    client, p = async_client_with_db, prediction_service
    resp = await client.post(
        f"/v1/services/{p.name}/predictions", json={"input": {}},
        headers={"Authorization": f"Bearer {p.raw_key}", "Prefer": "respond-async"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "id" in body and body["service"] == "pred-svc"
    assert body["status"] in ("starting", "processing", "succeeded")


@pytest.mark.asyncio
async def test_predictions_async_poll_to_succeeded(async_client_with_db, prediction_service):
    """async create → poll GET /predictions/{id} → succeeded + output。"""
    client, p = async_client_with_db, prediction_service
    resp = await client.post(
        f"/v1/services/{p.name}/predictions", json={"input": {}},
        headers={"Authorization": f"Bearer {p.raw_key}", "Prefer": "respond-async"})
    pid = resp.json()["id"]
    final = await _poll_until_done(client, pid)
    assert final["status"] == "completed"
    # 经 /predictions/{id} 也拿到终态
    pr = await client.get(f"/v1/predictions/{pid}",
                          headers={"Authorization": f"Bearer {p.raw_key}"})
    assert pr.status_code == 200 and pr.json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_predictions_sync_blocks_to_terminal(async_client_with_db, prediction_service):
    """无 Prefer(同步)→ 阻塞至终态 → 200 + status succeeded(inline text 工作流秒完成)。"""
    client, p = async_client_with_db, prediction_service
    resp = await client.post(
        f"/v1/services/{p.name}/predictions", json={"input": {"text": "hi"}},
        headers={"Authorization": f"Bearer {p.raw_key}"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_predictions_input_validation_422(async_client_with_db, prediction_service):
    """input 类型不符 schema → 422(PR-1 校验接进调用路径)。"""
    client, p = async_client_with_db, prediction_service
    resp = await client.post(
        f"/v1/services/{p.name}/predictions", json={"input": {"text": 123}},  # text 应 string
        headers={"Authorization": f"Bearer {p.raw_key}", "Prefer": "respond-async"})
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_predictions_sse_stream(async_client_with_db, prediction_service):
    """PR-3:同步跑完后 GET /predictions/{id}/stream → SSE 推一条 succeeded data 后结束。"""
    client, p = async_client_with_db, prediction_service
    # 同步跑(终态)
    resp = await client.post(
        f"/v1/services/{p.name}/predictions", json={"input": {}},
        headers={"Authorization": f"Bearer {p.raw_key}"})
    pid = resp.json()["id"]
    # 流(已终态 → 立即一条 + 结束)
    body = ""
    async with client.stream("GET", f"/v1/predictions/{pid}/stream",
                             headers={"Authorization": f"Bearer {p.raw_key}"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        async for chunk in r.aiter_text():
            body += chunk
            if "succeeded" in body:
                break
    assert "data:" in body and "succeeded" in body


@pytest.mark.asyncio
async def test_predictions_webhook_stored(async_client_with_db, prediction_service):
    """PR-3:请求带 webhook → 持久化到 ExecutionTask.webhook_url(供 runner 终态投递)。"""
    from src.models.execution_task import ExecutionTask
    client, p = async_client_with_db, prediction_service
    resp = await client.post(
        f"/v1/services/{p.name}/predictions",
        json={"input": {}, "webhook": "https://hook.example/cb", "webhook_events_filter": ["completed"]},
        headers={"Authorization": f"Bearer {p.raw_key}", "Prefer": "respond-async"})
    pid = int(resp.json()["id"])
    async with async_client_with_db.session_factory() as s:
        task = await s.get(ExecutionTask, pid)
        assert task.webhook_url == "https://hook.example/cb"
        assert task.webhook_events == ["completed"]
