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

    # Pin workflow_runner's session factory to the test DB.
    from src.services import workflow_runner as _runner
    monkeypatch.setattr(_runner, "create_session_factory",
                        lambda: session_factory)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.session_factory = session_factory  # type: ignore[attr-defined]
        yield c

    await engine.dispose()


@pytest.fixture
async def published_workflow_instance(async_client_with_db):
    """A source_type=workflow ServiceInstance with an inline-only workflow + API key."""
    from src.models.instance_api_key import InstanceApiKey
    from src.models.service_instance import ServiceInstance

    session_factory = async_client_with_db.session_factory
    raw = f"sk-test-{_secrets.token_hex(8)}"
    async with session_factory() as s:
        inst = ServiceInstance(
            source_type="workflow",
            source_name="inline-wf",
            name="Lane S e2e",
            type="workflow",
            status="active",
            params_override={
                "nodes": [
                    {"id": "t1", "type": "text_input",
                     "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
                ],
                "edges": [],
            },
        )
        s.add(inst)
        await s.commit()
        await s.refresh(inst)
        key = InstanceApiKey(
            instance_id=inst.id,
            label="test",
            key_hash=bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode(),
            key_prefix=raw[:10],
            is_active=True,
        )
        s.add(key)
        await s.commit()
        await s.refresh(key)

    class _Wrap:
        pass
    w = _Wrap()
    w.instance_id = inst.id
    w.raw_key = raw
    return w


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


@pytest.mark.asyncio
async def test_instance_run_returns_202(async_client_with_db, published_workflow_instance):
    """POST /v1/instances/{id}/run → 202 + task_id。"""
    client = async_client_with_db
    pwi = published_workflow_instance
    resp = await client.post(
        f"/v1/instances/{pwi.instance_id}/run",
        json={"inputs": {}},
        headers={"Authorization": f"Bearer {pwi.raw_key}"},
    )
    assert resp.status_code == 202, resp.text
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_instance_run_enqueue_poll_result(async_client_with_db, published_workflow_instance):
    """端到端：instance /run enqueue → poll → completed。"""
    client = async_client_with_db
    pwi = published_workflow_instance
    resp = await client.post(
        f"/v1/instances/{pwi.instance_id}/run",
        json={"inputs": {}},
        headers={"Authorization": f"Bearer {pwi.raw_key}"},
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    final = await _poll_until_done(client, task_id)
    assert final["status"] == "completed"
