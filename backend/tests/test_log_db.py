import os
import pytest
from src.services.log_db import init_log_db, insert_request_log, insert_app_log, insert_frontend_log, insert_audit_log, query_logs, get_log_db_path


@pytest.fixture
def log_db(tmp_path):
    db_path = str(tmp_path / "test_logs.db")
    init_log_db(db_path)
    return db_path


def test_init_creates_tables(log_db):
    import sqlite3
    conn = sqlite3.connect(log_db)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}
    assert "request_logs" in table_names
    assert "app_logs" in table_names
    assert "frontend_logs" in table_names
    assert "audit_logs" in table_names
    conn.close()


def test_insert_and_query_request_log(log_db):
    insert_request_log(log_db, method="GET", path="/api/v1/tasks", status=200, duration_ms=42, ip="127.0.0.1", user_agent="test")
    result = query_logs(log_db, "request_logs", limit=10)
    assert result["total"] >= 1
    assert result["items"][0]["method"] == "GET"
    assert result["items"][0]["status"] == 200


def test_insert_and_query_app_log(log_db):
    insert_app_log(log_db, level="ERROR", module="src.api.main", message="test error", location="main.py:42")
    result = query_logs(log_db, "app_logs", limit=10)
    assert result["total"] >= 1
    assert result["items"][0]["level"] == "ERROR"


def test_insert_and_query_frontend_log(log_db):
    insert_frontend_log(log_db, type="network", message="Request failed", page="/models", stack=None)
    result = query_logs(log_db, "frontend_logs", limit=10)
    assert result["total"] >= 1
    assert result["items"][0]["type"] == "network"


def test_insert_and_query_audit_log(log_db):
    insert_audit_log(log_db, action="load_engine", path="/api/v1/engines/cosyvoice2/load", method="POST", ip="127.0.0.1", detail='{}')
    result = query_logs(log_db, "audit_logs", limit=10)
    assert result["total"] >= 1
    assert result["items"][0]["action"] == "load_engine"


def test_query_with_search(log_db):
    insert_request_log(log_db, method="GET", path="/api/v1/engines", status=200, duration_ms=10, ip="127.0.0.1", user_agent="test")
    insert_request_log(log_db, method="POST", path="/api/v1/workflows", status=201, duration_ms=20, ip="127.0.0.1", user_agent="test")
    result = query_logs(log_db, "request_logs", search="engines", limit=10)
    assert result["total"] == 1
    assert "engines" in result["items"][0]["path"]


def test_query_with_since(log_db):
    insert_request_log(log_db, method="GET", path="/test", status=200, duration_ms=5, ip="127.0.0.1", user_agent="test")
    # Query with future timestamp should return 0
    result = query_logs(log_db, "request_logs", since="2099-01-01T00:00:00", limit=10)
    assert result["total"] == 0
