import pytest
from src.services.log_db import init_log_db, insert_request_log, insert_app_log


@pytest.fixture
async def log_client(tmp_path):
    db_path = str(tmp_path / "test_logs.db")
    init_log_db(db_path)
    # Insert some test data
    insert_request_log(db_path, method="GET", path="/api/v1/tasks", status=200, duration_ms=42, ip="127.0.0.1", user_agent="test")
    insert_request_log(db_path, method="POST", path="/api/v1/workflows", status=201, duration_ms=100, ip="127.0.0.1", user_agent="test")
    insert_app_log(db_path, level="ERROR", module="test", message="boom", location="test.py:1")

    from httpx import ASGITransport, AsyncClient
    from src.api.main import create_app
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_query_request_logs(log_client):
    resp = await log_client.get("/api/v1/logs/requests")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "items" in data


async def test_query_app_logs(log_client):
    resp = await log_client.get("/api/v1/logs/app")
    assert resp.status_code == 200


async def test_query_frontend_logs(log_client):
    resp = await log_client.get("/api/v1/logs/frontend")
    assert resp.status_code == 200


async def test_report_frontend_log(log_client):
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


async def test_query_with_limit(log_client):
    resp = await log_client.get("/api/v1/logs/requests?limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) <= 1
