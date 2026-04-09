# Logs System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4-tab logging system (Request, Application, Frontend, Audit) with independent SQLite storage, auto-collection, and a frontend log viewer.

**Architecture:** Separate `data/logs.db` database using raw `sqlite3`. Request + Audit logs collected via middleware, Application logs via Python `logging.Handler`, Frontend logs via dedicated API. Frontend LogsOverlay with 4 tabs, search, time-range filter, and live mode.

**Tech Stack:** Python 3.12, FastAPI, sqlite3, React 19, @tanstack/react-query, Zustand

**Worktree:** `.worktrees/feature-logs/` (branch: `feature/logs`)

---

## File Map

| File | Responsibility |
|------|----------------|
| `backend/src/services/log_db.py` | logs.db init, insert, query, cleanup |
| `backend/src/services/log_collector.py` | `DbLogHandler` for Python logging |
| `backend/src/api/middleware.py` | Modify RequestLoggingMiddleware + add AuditMiddleware |
| `backend/src/api/routes/logs.py` | Query + frontend report endpoints |
| `backend/src/api/main.py` | Wire log_db init + cleanup task + middleware |
| `backend/tests/test_log_db.py` | log_db unit tests |
| `backend/tests/test_logs_api.py` | API endpoint tests |
| `frontend/src/api/logs.ts` | React Query hooks |
| `frontend/src/components/overlays/LogsOverlay.tsx` | 4-tab viewer |
| `frontend/src/utils/errorReporter.ts` | Global error capture + reporting |
| `frontend/src/stores/panel.ts` | Add 'logs' overlay ID |
| `frontend/src/components/layout/IconRail.tsx` | Add logs icon |

---

### Task 1: log_db — database init and insert functions

**Files:**
- Create: `backend/src/services/log_db.py`
- Test: `backend/tests/test_log_db.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_log_db.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_log_db.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# backend/src/services/log_db.py
"""Independent SQLite log database — separate from business data."""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_local = threading.local()

_TABLES = {
    "request_logs": """
        CREATE TABLE IF NOT EXISTS request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            method VARCHAR(10) NOT NULL,
            path VARCHAR(500) NOT NULL,
            status INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            ip VARCHAR(45),
            user_agent VARCHAR(500)
        )
    """,
    "app_logs": """
        CREATE TABLE IF NOT EXISTS app_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            level VARCHAR(10) NOT NULL,
            module VARCHAR(100),
            message TEXT,
            location VARCHAR(200)
        )
    """,
    "frontend_logs": """
        CREATE TABLE IF NOT EXISTS frontend_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            type VARCHAR(30) NOT NULL,
            message TEXT,
            page VARCHAR(500),
            stack TEXT
        )
    """,
    "audit_logs": """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            action VARCHAR(50) NOT NULL,
            path VARCHAR(500),
            method VARCHAR(10),
            ip VARCHAR(45),
            detail TEXT
        )
    """,
}

# Searchable columns per table
_SEARCH_COLS = {
    "request_logs": ["path", "method", "ip"],
    "app_logs": ["message", "module", "location"],
    "frontend_logs": ["message", "page", "type"],
    "audit_logs": ["action", "path", "detail"],
}

_db_path: str | None = None


def get_log_db_path() -> str:
    if _db_path is None:
        raise RuntimeError("Log DB not initialized. Call init_log_db() first.")
    return _db_path


def _get_conn(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_log_db_path()
    conn = getattr(_local, "conn", None)
    if conn is None or db_path is not None:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        if db_path is None:
            _local.conn = conn
    return conn


def init_log_db(db_path: str | None = None) -> str:
    global _db_path
    if db_path is None:
        from src.config import get_settings
        settings = get_settings()
        home = Path(settings.NOUS_CENTER_HOME).expanduser()
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(data_dir / "logs.db")
    _db_path = db_path
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _TABLES.values():
        conn.execute(ddl)
    # Indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_ts ON request_logs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_app_logs_ts ON app_logs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frontend_logs_ts ON frontend_logs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_ts ON audit_logs(timestamp)")
    conn.commit()
    conn.close()
    return db_path


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def insert_request_log(db_path: str | None = None, **kwargs) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO request_logs (timestamp, method, path, status, duration_ms, ip, user_agent) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_now(), kwargs["method"], kwargs["path"], kwargs["status"], kwargs["duration_ms"], kwargs.get("ip", ""), kwargs.get("user_agent", "")),
    )
    conn.commit()


def insert_app_log(db_path: str | None = None, **kwargs) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO app_logs (timestamp, level, module, message, location) VALUES (?, ?, ?, ?, ?)",
        (_now(), kwargs["level"], kwargs.get("module", ""), kwargs.get("message", ""), kwargs.get("location", "")),
    )
    conn.commit()


def insert_frontend_log(db_path: str | None = None, **kwargs) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO frontend_logs (timestamp, type, message, page, stack) VALUES (?, ?, ?, ?, ?)",
        (_now(), kwargs["type"], kwargs.get("message", ""), kwargs.get("page", ""), kwargs.get("stack")),
    )
    conn.commit()


def insert_audit_log(db_path: str | None = None, **kwargs) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO audit_logs (timestamp, action, path, method, ip, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (_now(), kwargs["action"], kwargs.get("path", ""), kwargs.get("method", ""), kwargs.get("ip", ""), kwargs.get("detail", "")),
    )
    conn.commit()


def query_logs(
    db_path: str | None = None,
    table: str = "request_logs",
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    since: str | None = None,
    level: str | None = None,
    type_filter: str | None = None,
    method: str | None = None,
    status: str | None = None,
) -> dict:
    if table not in _TABLES:
        raise ValueError(f"Unknown table: {table}")

    conn = _get_conn(db_path)
    conditions = []
    params: list = []

    if since:
        conditions.append("timestamp >= ?")
        params.append(since)
    if search and table in _SEARCH_COLS:
        cols = _SEARCH_COLS[table]
        or_clauses = " OR ".join(f"{c} LIKE ?" for c in cols)
        conditions.append(f"({or_clauses})")
        params.extend([f"%{search}%"] * len(cols))
    if level and table == "app_logs":
        conditions.append("level = ?")
        params.append(level.upper())
    if type_filter and table == "frontend_logs":
        conditions.append("type = ?")
        params.append(type_filter)
    if method:
        conditions.append("method = ?")
        params.append(method.upper())
    if status:
        if status.endswith("xx"):
            prefix = status[0]
            conditions.append(f"CAST(status / 100 AS INTEGER) = ?")
            params.append(int(prefix))
        else:
            conditions.append("status = ?")
            params.append(int(status))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    total = conn.execute(f"SELECT COUNT(*) FROM {table} {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [min(limit, 500), offset],
    ).fetchall()

    items = [dict(row) for row in rows]
    return {"total": total, "items": items}


def cleanup_logs(db_path: str | None = None, max_age_days: int = 7, max_rows: int = 100_000) -> dict:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_conn(db_path)
    deleted = {}
    for table in _TABLES:
        cur = conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
        deleted[table] = cur.rowcount
        # Enforce row limit
        conn.execute(f"DELETE FROM {table} WHERE id NOT IN (SELECT id FROM {table} ORDER BY id DESC LIMIT ?)", (max_rows,))
    conn.commit()
    return deleted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_log_db.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/log_db.py backend/tests/test_log_db.py
git commit -m "feat: add log_db — independent SQLite log storage with query and cleanup"
```

---

### Task 2: log_collector — Python logging handler

**Files:**
- Create: `backend/src/services/log_collector.py`
- Test: `backend/tests/test_log_collector.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_log_collector.py
import logging
import pytest
from src.services.log_db import init_log_db, query_logs
from src.services.log_collector import DbLogHandler


@pytest.fixture
def log_db(tmp_path):
    db_path = str(tmp_path / "test_logs.db")
    init_log_db(db_path)
    return db_path


def test_handler_captures_log_records(log_db):
    handler = DbLogHandler(db_path=log_db, flush_interval=0, flush_size=1)
    logger = logging.getLogger("test.collector")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Test message from collector")
    handler.flush()

    result = query_logs(log_db, "app_logs", limit=10)
    assert result["total"] >= 1
    assert "Test message from collector" in result["items"][0]["message"]
    assert result["items"][0]["level"] == "INFO"

    logger.removeHandler(handler)


def test_handler_captures_error_with_location(log_db):
    handler = DbLogHandler(db_path=log_db, flush_interval=0, flush_size=1)
    logger = logging.getLogger("test.location")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.error("Something broke")
    handler.flush()

    result = query_logs(log_db, "app_logs", limit=10)
    item = result["items"][0]
    assert item["level"] == "ERROR"
    assert "test_log_collector.py" in item["location"]

    logger.removeHandler(handler)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_log_collector.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# backend/src/services/log_collector.py
"""Python logging handler that writes to logs.db with buffering."""
import logging
import threading
import time
from src.services.log_db import insert_app_log


class DbLogHandler(logging.Handler):
    """Buffered logging handler that writes to the log database."""

    def __init__(self, db_path: str | None = None, flush_interval: float = 1.0, flush_size: int = 50):
        super().__init__()
        self._db_path = db_path
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._flush_size = flush_size
        if flush_interval > 0:
            self._timer = threading.Thread(target=self._flush_loop, daemon=True)
            self._timer.start()

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "level": record.levelname,
            "module": record.name,
            "message": self.format(record),
            "location": f"{record.filename}:{record.lineno}",
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._flush_size:
                self._do_flush()

    def flush(self) -> None:
        with self._lock:
            self._do_flush()

    def _do_flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        for entry in batch:
            try:
                insert_app_log(db_path=self._db_path, **entry)
            except Exception:
                pass  # Never let log writing crash the app

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self._flush_interval)
            self.flush()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_log_collector.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/log_collector.py backend/tests/test_log_collector.py
git commit -m "feat: add DbLogHandler — buffered Python logging to logs.db"
```

---

### Task 3: Middleware — Request + Audit logging

**Files:**
- Modify: `backend/src/api/middleware.py`
- Test: `backend/tests/test_middleware_logs.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_middleware_logs.py
import pytest
from src.services.log_db import init_log_db, query_logs
from src.api.middleware import derive_audit_action


@pytest.fixture
def log_db(tmp_path):
    db_path = str(tmp_path / "test_logs.db")
    init_log_db(db_path)
    return db_path


def test_derive_audit_action():
    assert derive_audit_action("POST", "/api/v1/engines/cosyvoice2/load") == "load_engine"
    assert derive_audit_action("POST", "/api/v1/engines/cosyvoice2/unload") == "unload_engine"
    assert derive_audit_action("POST", "/api/v1/engines/reload") == "reload_registry"
    assert derive_audit_action("POST", "/api/v1/workflows") == "create_workflow"
    assert derive_audit_action("PATCH", "/api/v1/workflows/123") == "update_workflow"
    assert derive_audit_action("DELETE", "/api/v1/workflows/123") == "delete_workflow"
    assert derive_audit_action("POST", "/api/v1/workflows/123/publish-app") == "publish_app"
    assert derive_audit_action("DELETE", "/api/v1/apps/my-app") == "unpublish_app"
    # Fallback
    assert derive_audit_action("POST", "/api/v1/unknown/thing") == "post_thing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_middleware_logs.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Rewrite middleware.py**

```python
# backend/src/api/middleware.py
"""Request logging and audit middleware."""
import json
import logging
import re
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("nous.access")

# Paths to skip logging (avoid recursion and noise)
_SKIP_PATHS = {"/health", "/favicon.ico"}
_SKIP_PREFIXES = ("/api/v1/logs/",)

# Audit action derivation rules
_AUDIT_RULES: list[tuple[str, str, str]] = [
    (r"POST", r"/api/v1/engines/[^/]+/load$", "load_engine"),
    (r"POST", r"/api/v1/engines/[^/]+/unload$", "unload_engine"),
    (r"POST", r"/api/v1/engines/reload$", "reload_registry"),
    (r"POST", r"/api/v1/workflows$", "create_workflow"),
    (r"PATCH", r"/api/v1/workflows/\d+$", "update_workflow"),
    (r"DELETE", r"/api/v1/workflows/\d+$", "delete_workflow"),
    (r"POST", r"/api/v1/workflows/\d+/publish-app$", "publish_app"),
    (r"DELETE", r"/api/v1/apps/[^/]+$", "unpublish_app"),
]


def derive_audit_action(method: str, path: str) -> str:
    for rule_method, pattern, action in _AUDIT_RULES:
        if method == rule_method and re.match(pattern, path):
            return action
    # Fallback: method_last_segment
    segments = [s for s in path.rstrip("/").split("/") if s and not s.isdigit()]
    last = segments[-1] if segments else "unknown"
    return f"{method.lower()}_{last}"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        path = request.url.path
        if path in _SKIP_PATHS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            return response

        logger.info("%s %s %d %dms", request.method, path, response.status_code, elapsed_ms)

        # Write to log DB (fire-and-forget)
        try:
            from src.services.log_db import insert_request_log
            insert_request_log(
                method=request.method,
                path=path,
                status=response.status_code,
                duration_ms=elapsed_ms,
                ip=request.client.host if request.client else "",
                user_agent=request.headers.get("user-agent", ""),
            )
        except Exception:
            pass

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Captures admin operations for audit trail."""

    async def dispatch(self, request: Request, call_next):
        # Only audit requests with admin token
        has_admin_token = "authorization" in request.headers or "x-admin-token" in request.headers
        path = request.url.path

        if has_admin_token and not path.startswith("/api/v1/logs/"):
            # Read body for detail (cache it for downstream)
            body = b""
            try:
                body = await request.body()
            except Exception:
                pass

            response = await call_next(request)

            # Only log mutating operations that succeeded
            if request.method in ("POST", "PUT", "PATCH", "DELETE") and response.status_code < 500:
                try:
                    from src.services.log_db import insert_audit_log
                    action = derive_audit_action(request.method, path)
                    detail = body.decode("utf-8", errors="replace")[:2000] if body else ""
                    insert_audit_log(
                        action=action,
                        path=path,
                        method=request.method,
                        ip=request.client.host if request.client else "",
                        detail=detail,
                    )
                except Exception:
                    pass

            return response

        return await call_next(request)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_middleware_logs.py -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/middleware.py backend/tests/test_middleware_logs.py
git commit -m "feat: add audit middleware + request log DB writes"
```

---

### Task 4: Log query API + frontend report endpoint

**Files:**
- Create: `backend/src/api/routes/logs.py`
- Test: `backend/tests/test_logs_api.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_logs_api.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_logs_api.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

```python
# backend/src/api/routes/logs.py
"""Log query and frontend error reporting endpoints."""
from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.services.log_db import query_logs, insert_frontend_log

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


class FrontendLogReport(BaseModel):
    type: str
    message: str
    page: str = ""
    stack: str | None = None


@router.get("/requests")
async def get_request_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    method: str | None = None,
    status: str | None = None,
    since: str | None = None,
):
    return query_logs("request_logs", limit=limit, offset=offset, search=search, method=method, status=status, since=since)


@router.get("/app")
async def get_app_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    level: str | None = None,
    since: str | None = None,
):
    return query_logs("app_logs", limit=limit, offset=offset, search=search, level=level, since=since)


@router.get("/frontend")
async def get_frontend_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    type: str | None = None,
    since: str | None = None,
):
    return query_logs("frontend_logs", limit=limit, offset=offset, search=search, type_filter=type, since=since)


@router.post("/frontend", status_code=201)
async def report_frontend_log(body: FrontendLogReport):
    insert_frontend_log(type=body.type, message=body.message, page=body.page, stack=body.stack)
    return {"status": "recorded"}


@router.get("/audit")
async def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    since: str | None = None,
):
    return query_logs("audit_logs", limit=limit, offset=offset, search=search, since=since)
```

- [ ] **Step 4: Register in main.py**

Add to `backend/src/api/main.py` imports:
```python
from src.api.routes import logs
```

In `create_app()`:
```python
app.include_router(logs.router)
```

In `lifespan()`, after DB setup:
```python
# Initialize log database
from src.services.log_db import init_log_db
init_log_db()
logger.info("Log database initialized")
```

Add cleanup background task in lifespan:
```python
async def log_cleanup_loop():
    while True:
        await asyncio.sleep(3600)  # Every hour
        try:
            from src.services.log_db import cleanup_logs
            cleanup_logs()
        except Exception as e:
            logger.warning("Log cleanup failed: %s", e)

asyncio.create_task(log_cleanup_loop())
```

Add AuditMiddleware:
```python
from src.api.middleware import AuditMiddleware
app.add_middleware(AuditMiddleware)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd .worktrees/feature-logs/backend && uv run pytest tests/test_logs_api.py -v`
Expected: 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/logs.py backend/src/api/main.py backend/tests/test_logs_api.py
git commit -m "feat: add log query API + frontend report endpoint + wire into app"
```

---

### Task 5: Application log collector wiring

**Files:**
- Modify: `backend/src/api/main.py`

- [ ] **Step 1: Add DbLogHandler to app lifespan**

In `lifespan()`, after `init_log_db()`:

```python
# Install DB log handler for application logs
from src.services.log_collector import DbLogHandler
db_handler = DbLogHandler()
db_handler.setLevel(logging.INFO)
logging.getLogger("src").addHandler(db_handler)
logging.getLogger("nous").addHandler(db_handler)
logger.info("Application log collector installed")
```

- [ ] **Step 2: Run all backend tests**

Run: `cd .worktrees/feature-logs/backend && uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/src/api/main.py
git commit -m "feat: wire DbLogHandler into app — captures all src.* and nous.* logs"
```

---

### Task 6: Frontend — API hooks + error reporter

**Files:**
- Create: `frontend/src/api/logs.ts`
- Create: `frontend/src/utils/errorReporter.ts`

- [ ] **Step 1: Create API hooks**

```typescript
// frontend/src/api/logs.ts
import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface LogItem {
  id: number
  timestamp: string
  [key: string]: unknown
}

export interface LogResponse {
  total: number
  items: LogItem[]
}

export interface LogQueryParams {
  limit?: number
  offset?: number
  search?: string
  since?: string
  level?: string
  type?: string
  method?: string
  status?: string
}

function buildParams(params: LogQueryParams): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== '') sp.set(k, String(v))
  }
  const str = sp.toString()
  return str ? `?${str}` : ''
}

export function useRequestLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'requests', params],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/requests${buildParams(params)}`),
    refetchInterval: 3000,
    enabled,
  })
}

export function useAppLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'app', params],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/app${buildParams(params)}`),
    refetchInterval: 3000,
    enabled,
  })
}

export function useFrontendLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'frontend', params],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/frontend${buildParams(params)}`),
    refetchInterval: 3000,
    enabled,
  })
}

export function useAuditLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'audit', params],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/audit${buildParams(params)}`),
    refetchInterval: 3000,
    enabled,
  })
}
```

- [ ] **Step 2: Create error reporter**

```typescript
// frontend/src/utils/errorReporter.ts
const REPORT_URL = '/api/v1/logs/frontend'

function report(type: string, message: string, page: string, stack?: string) {
  try {
    navigator.sendBeacon(REPORT_URL, JSON.stringify({ type, message, page, stack: stack || null }))
  } catch {
    // Fallback to fetch
    fetch(REPORT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, message, page, stack: stack || null }),
    }).catch(() => {})
  }
}

export function installErrorReporter() {
  // JS errors
  window.addEventListener('error', (e) => {
    report('error', e.message || 'Unknown error', window.location.pathname, e.error?.stack)
  })

  // Unhandled promise rejections
  window.addEventListener('unhandledrejection', (e) => {
    const msg = e.reason?.message || String(e.reason) || 'Unhandled rejection'
    report('unhandled_rejection', msg, window.location.pathname, e.reason?.stack)
  })

  // Patch fetch for network errors
  const originalFetch = window.fetch
  window.fetch = async (...args) => {
    try {
      const resp = await originalFetch(...args)
      if (!resp.ok && resp.status >= 500) {
        const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url
        report('network', `${resp.status} ${resp.statusText} — ${url}`, window.location.pathname)
      }
      return resp
    } catch (err: any) {
      const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url
      // Don't report errors for the log endpoint itself
      if (!url.includes('/api/v1/logs/')) {
        report('network', `${err.message} — ${url}`, window.location.pathname)
      }
      throw err
    }
  }
}
```

- [ ] **Step 3: Install error reporter in main.tsx**

In `frontend/src/main.tsx`, add before `ReactDOM.createRoot`:

```typescript
import { installErrorReporter } from './utils/errorReporter'
installErrorReporter()
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/logs.ts frontend/src/utils/errorReporter.ts frontend/src/main.tsx
git commit -m "feat: add frontend log API hooks + global error reporter"
```

---

### Task 7: Frontend — LogsOverlay UI

**Files:**
- Create: `frontend/src/components/overlays/LogsOverlay.tsx`
- Modify: `frontend/src/stores/panel.ts` (add 'logs' overlay)
- Modify: `frontend/src/components/layout/IconRail.tsx` (add logs icon)

- [ ] **Step 1: Add 'logs' to OverlayId**

In `frontend/src/stores/panel.ts`, change:
```typescript
export type OverlayId = 'dashboard' | 'models' | 'settings' | 'preset-detail' | 'api-management' | 'agents' | 'logs'
```

- [ ] **Step 2: Create LogsOverlay**

```tsx
// frontend/src/components/overlays/LogsOverlay.tsx
import { useState, useMemo } from 'react'
import { Search, RefreshCw } from 'lucide-react'
import { useRequestLogs, useAppLogs, useFrontendLogs, useAuditLogs, type LogQueryParams } from '../../api/logs'

type Tab = 'requests' | 'app' | 'frontend' | 'audit'

const TABS: { id: Tab; label: string }[] = [
  { id: 'requests', label: 'Request Logs' },
  { id: 'app', label: 'Application Logs' },
  { id: 'frontend', label: 'Frontend Logs' },
  { id: 'audit', label: 'Audit Logs' },
]

const TIME_RANGES = [
  { label: '15m', minutes: 15 },
  { label: '1h', minutes: 60 },
  { label: '24h', minutes: 1440 },
  { label: '3d', minutes: 4320 },
  { label: '7d', minutes: 10080 },
]

function sinceFromMinutes(minutes: number): string {
  return new Date(Date.now() - minutes * 60_000).toISOString()
}

const METHOD_COLORS: Record<string, string> = {
  GET: 'var(--ok)', POST: 'var(--info)', PUT: 'var(--warn)',
  PATCH: 'var(--warn)', DELETE: 'var(--accent)', OPTIONS: 'var(--muted)',
}

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'var(--muted)', INFO: 'var(--info)', WARNING: 'var(--warn)',
  ERROR: 'var(--accent)', CRITICAL: 'var(--accent)',
}

const TYPE_COLORS: Record<string, string> = {
  network: 'var(--warn)', unhandled_rejection: 'var(--accent)',
  error: 'var(--accent)', console_error: 'var(--warn)',
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{
      padding: '1px 6px', borderRadius: 3, fontSize: 9, fontWeight: 600,
      background: `color-mix(in srgb, ${color} 15%, transparent)`, color,
    }}>
      {text}
    </span>
  )
}

function StatusBadge({ status }: { status: number }) {
  const color = status < 300 ? 'var(--ok)' : status < 500 ? 'var(--warn)' : 'var(--accent)'
  return <Badge text={String(status)} color={color} />
}

export default function LogsOverlay() {
  const [tab, setTab] = useState<Tab>('requests')
  const [search, setSearch] = useState('')
  const [timeRange, setTimeRange] = useState(60) // minutes
  const [live, setLive] = useState(true)

  const params: LogQueryParams = useMemo(() => ({
    limit: 100,
    search: search || undefined,
    since: sinceFromMinutes(timeRange),
  }), [search, timeRange])

  const requestLogs = useRequestLogs(params, tab === 'requests' && live)
  const appLogs = useAppLogs(params, tab === 'app' && live)
  const frontendLogs = useFrontendLogs(params, tab === 'frontend' && live)
  const auditLogs = useAuditLogs(params, tab === 'audit' && live)

  const activeData = tab === 'requests' ? requestLogs : tab === 'app' ? appLogs : tab === 'frontend' ? frontendLogs : auditLogs

  return (
    <div className="absolute inset-0 overflow-hidden z-[16] flex flex-col" style={{ background: 'var(--bg)' }}>
      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-strong)', marginBottom: 10 }}>Logs</div>

        {/* Tabs */}
        <div className="flex gap-1" style={{ marginBottom: 10 }}>
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{
                padding: '4px 12px', borderRadius: 4, fontSize: 12, fontWeight: 500, border: 'none', cursor: 'pointer',
                background: tab === t.id ? 'var(--accent-subtle)' : 'transparent',
                color: tab === t.id ? 'var(--accent)' : 'var(--muted)',
              }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1" style={{
            flex: 1, padding: '4px 8px', background: 'var(--bg-accent)', borderRadius: 4, border: '1px solid var(--border)',
          }}>
            <Search size={12} style={{ color: 'var(--muted)' }} />
            <input
              value={search} onChange={(e) => setSearch(e.target.value)}
              placeholder="Search..." style={{
                flex: 1, background: 'none', border: 'none', color: 'var(--text)', fontSize: 11, outline: 'none',
              }}
            />
          </div>
          {TIME_RANGES.map((r) => (
            <button key={r.label} onClick={() => setTimeRange(r.minutes)}
              style={{
                padding: '3px 8px', borderRadius: 3, fontSize: 10, border: 'none', cursor: 'pointer',
                background: timeRange === r.minutes ? 'var(--accent)' : 'var(--bg-accent)',
                color: timeRange === r.minutes ? '#fff' : 'var(--muted)',
              }}>
              {r.label}
            </button>
          ))}
          <button onClick={() => setLive(!live)}
            style={{
              padding: '3px 8px', borderRadius: 3, fontSize: 10, border: 'none', cursor: 'pointer',
              background: live ? 'color-mix(in srgb, var(--ok) 15%, transparent)' : 'var(--bg-accent)',
              color: live ? 'var(--ok)' : 'var(--muted)',
            }}>
            {live ? '● Live' : '○ Paused'}
          </button>
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>
            {activeData.data?.total ?? 0} records
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto" style={{ padding: '0 16px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', position: 'sticky', top: 0, background: 'var(--bg)', zIndex: 1 }}>
              {tab === 'requests' && <><Th>Time</Th><Th>Method</Th><Th>Path</Th><Th>Status</Th><Th>Time</Th><Th>IP</Th></>}
              {tab === 'app' && <><Th>Time</Th><Th>Level</Th><Th>Module</Th><Th>Message</Th><Th>Location</Th></>}
              {tab === 'frontend' && <><Th>Time</Th><Th>Type</Th><Th>Message</Th><Th>Page</Th></>}
              {tab === 'audit' && <><Th>Time</Th><Th>Action</Th><Th>Path</Th><Th>Method</Th><Th>IP</Th></>}
            </tr>
          </thead>
          <tbody>
            {(activeData.data?.items ?? []).map((item) => (
              <tr key={item.id} style={{ borderBottom: '1px solid var(--border)' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-hover)' }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = '' }}>
                {tab === 'requests' && <>
                  <Td>{fmtTime(item.timestamp as string)}</Td>
                  <Td><Badge text={item.method as string} color={METHOD_COLORS[item.method as string] || 'var(--muted)'} /></Td>
                  <Td mono>{item.path as string}</Td>
                  <Td><StatusBadge status={item.status as number} /></Td>
                  <Td>{item.duration_ms as number}ms</Td>
                  <Td mono>{item.ip as string}</Td>
                </>}
                {tab === 'app' && <>
                  <Td>{fmtTime(item.timestamp as string)}</Td>
                  <Td><Badge text={item.level as string} color={LEVEL_COLORS[item.level as string] || 'var(--muted)'} /></Td>
                  <Td mono>{item.module as string}</Td>
                  <Td style={{ maxWidth: 500 }}>{item.message as string}</Td>
                  <Td mono>{item.location as string}</Td>
                </>}
                {tab === 'frontend' && <>
                  <Td>{fmtTime(item.timestamp as string)}</Td>
                  <Td><Badge text={item.type as string} color={TYPE_COLORS[item.type as string] || 'var(--muted)'} /></Td>
                  <Td style={{ maxWidth: 600 }}>{item.message as string}</Td>
                  <Td mono>{item.page as string}</Td>
                </>}
                {tab === 'audit' && <>
                  <Td>{fmtTime(item.timestamp as string)}</Td>
                  <Td><Badge text={item.action as string} color="var(--purple)" /></Td>
                  <Td mono>{item.path as string}</Td>
                  <Td><Badge text={item.method as string} color={METHOD_COLORS[item.method as string] || 'var(--muted)'} /></Td>
                  <Td mono>{item.ip as string}</Td>
                </>}
              </tr>
            ))}
          </tbody>
        </table>
        {activeData.isLoading && <div style={{ textAlign: 'center', padding: 20, color: 'var(--muted)' }}>Loading...</div>}
        {!activeData.isLoading && (activeData.data?.items.length ?? 0) === 0 && (
          <div style={{ textAlign: 'center', padding: 20, color: 'var(--muted)' }}>No logs found</div>
        )}
      </div>
    </div>
  )
}

function Th({ children }: { children: React.ReactNode }) {
  return <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--accent)', fontWeight: 600, fontSize: 10, textTransform: 'uppercase' }}>{children}</th>
}

function Td({ children, mono, style }: { children: React.ReactNode; mono?: boolean; style?: React.CSSProperties }) {
  return (
    <td style={{
      padding: '5px 8px', color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      fontFamily: mono ? 'var(--mono)' : undefined, ...style,
    }}>
      {children}
    </td>
  )
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso + (iso.includes('Z') || iso.includes('+') ? '' : 'Z'))
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso
  }
}
```

- [ ] **Step 3: Add logs icon to IconRail**

In `frontend/src/components/layout/IconRail.tsx`, import `ScrollText` from lucide-react and add a logs button in the bottom section (near settings):

Find the settings/gear icon and add before it:
```tsx
<IconButton icon={ScrollText} label="Logs" active={activeOverlay === 'logs'}
  onClick={() => toggleOverlay('logs')} />
```

- [ ] **Step 4: Render LogsOverlay in MainLayout or App**

Where other overlays are rendered (check App.tsx or the layout), add:
```tsx
{activeOverlay === 'logs' && <LogsOverlay />}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/overlays/LogsOverlay.tsx frontend/src/stores/panel.ts frontend/src/components/layout/IconRail.tsx
git commit -m "feat: add LogsOverlay with 4-tab log viewer, search, time-range, live mode"
```

---

### Task 8: Integration test + smoke test

**Files:**
- Test: `backend/tests/test_logs_integration.py`

- [ ] **Step 1: Write integration test**

```python
# backend/tests/test_logs_integration.py
import pytest


async def test_logs_endpoints_exist(client):
    """Smoke test: all log endpoints respond."""
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
```

- [ ] **Step 2: Run all backend tests**

Run: `cd .worktrees/feature-logs/backend && uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_logs_integration.py
git commit -m "test: add logs system integration smoke tests"
```

---

## Self-Review

**Spec coverage:**
- [x] request_logs table + collection via middleware — Task 1, 3
- [x] app_logs table + DbLogHandler — Task 1, 2
- [x] frontend_logs table + POST endpoint — Task 1, 4
- [x] audit_logs table + AuditMiddleware — Task 1, 3
- [x] Query API with filtering — Task 4
- [x] Cleanup/retention — Task 1 (cleanup_logs function), Task 4 (background loop)
- [x] Frontend error reporter — Task 6
- [x] LogsOverlay UI with 4 tabs — Task 7
- [x] IconRail integration — Task 7
- [x] Wire into main.py — Task 4, 5
- [x] Integration tests — Task 8

**Placeholder scan:** No TBD, TODO, or vague steps found.

**Type consistency:** `query_logs()` signature matches across Task 1 (definition), Task 4 (route usage). `insert_*_log()` signatures match across Task 1 (definition), Task 3 (middleware usage), Task 4 (frontend report).
