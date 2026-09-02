import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.main import create_app
from src.models.database import get_async_session
from src.models.log_entry import AppLog, RequestLog


@pytest.fixture
async def log_client(pg_engine):
    """App client backed by the temp PostgreSQL test DB with seeded log rows.
    (timestamp 列本来就是 String(32) 的 CST 字符串,见 log_entry.py —— 不是 datetime。)"""
    engine = pg_engine
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as s:
        s.add_all([
            RequestLog(timestamp="2026-06-10 10:00:00", method="GET", path="/api/v1/tasks", status=200, duration_ms=42, ip="127.0.0.1", user_agent="test"),
            RequestLog(timestamp="2026-06-10 10:00:01", method="POST", path="/api/v1/workflows", status=201, duration_ms=100, ip="127.0.0.1", user_agent="test"),
            AppLog(timestamp="2026-06-10 10:00:02", level="ERROR", module="test", message="boom", location="test.py:1"),
        ])
        await s.commit()

    async def override_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_async_session] = override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()


async def test_query_request_logs(log_client):
    resp = await log_client.get("/api/v1/logs/requests")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


async def test_query_app_logs(log_client):
    resp = await log_client.get("/api/v1/logs/app")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


async def test_query_frontend_logs(log_client):
    resp = await log_client.get("/api/v1/logs/frontend")
    assert resp.status_code == 200


async def test_report_frontend_log(log_client):
    # enqueue() drops silently when the writer isn't running (no lifespan here);
    # the endpoint still acknowledges with 201.
    resp = await log_client.post("/api/v1/logs/frontend", json={
        "type": "network",
        "message": "GET /api/v1/search — Request failed",
        "page": "/models",
    })
    assert resp.status_code == 201


async def test_query_audit_logs(log_client):
    resp = await log_client.get("/api/v1/logs/audit")
    assert resp.status_code == 200


async def test_query_with_search(log_client):
    resp = await log_client.get("/api/v1/logs/requests?search=tasks")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


async def test_query_with_limit(log_client):
    resp = await log_client.get("/api/v1/logs/requests?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1
