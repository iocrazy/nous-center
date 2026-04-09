import pytest


async def test_logs_endpoints_exist(client):
    for path in ["/api/v1/logs/requests", "/api/v1/logs/app", "/api/v1/logs/frontend", "/api/v1/logs/audit"]:
        resp = await client.get(path)
        assert resp.status_code == 200, f"{path} failed with {resp.status_code}"


async def test_frontend_log_report(client):
    resp = await client.post("/api/v1/logs/frontend", json={
        "type": "error",
        "message": "Test error",
        "page": "/test",
    })
    assert resp.status_code == 201


def test_all_log_imports():
    from src.services.log_db import init_log_db, insert_request_log, insert_app_log, insert_frontend_log, insert_audit_log, query_logs, cleanup_logs
    from src.services.log_collector import DbLogHandler
    from src.api.middleware import RequestLoggingMiddleware, AuditMiddleware, derive_audit_action
    assert True
