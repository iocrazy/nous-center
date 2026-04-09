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
            conditions.append("CAST(status / 100 AS INTEGER) = ?")
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
