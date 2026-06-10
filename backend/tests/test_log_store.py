"""PR-1: PG-backed log store (query/cleanup + async queue writer). Runs against
the conftest SQLite test DB; the log models register on Base via this import."""
import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.log_entry import AppLog, AuditLog, FrontendLog, RequestLog
from src.services.log_store import LogWriter, cleanup_logs, query_logs


async def _make_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/log_store.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


# ---- query_logs ----

async def test_query_request_logs_basic(db_session):
    db_session.add_all([
        RequestLog(timestamp="2026-06-10 10:00:00", method="GET", path="/api/v1/engines", status=200, duration_ms=10, ip="1.1.1.1", user_agent="t"),
        RequestLog(timestamp="2026-06-10 10:00:01", method="POST", path="/api/v1/workflows", status=201, duration_ms=20, ip="1.1.1.1", user_agent="t"),
    ])
    await db_session.commit()

    res = await query_logs(db_session, "request_logs", limit=10)
    assert res["total"] == 2
    # newest (highest id) first
    assert res["items"][0]["path"] == "/api/v1/workflows"
    assert set(res["items"][0]) >= {"id", "timestamp", "method", "path", "status", "duration_ms"}


async def test_query_search_and_method_and_status(db_session):
    db_session.add_all([
        RequestLog(timestamp="2026-06-10 10:00:00", method="GET", path="/api/v1/engines", status=200, duration_ms=10),
        RequestLog(timestamp="2026-06-10 10:00:01", method="POST", path="/api/v1/workflows", status=500, duration_ms=20),
    ])
    await db_session.commit()

    assert (await query_logs(db_session, "request_logs", search="engines"))["total"] == 1
    assert (await query_logs(db_session, "request_logs", method="post"))["total"] == 1
    assert (await query_logs(db_session, "request_logs", status="5xx"))["total"] == 1
    assert (await query_logs(db_session, "request_logs", status="200"))["total"] == 1


async def test_query_since_and_app_level(db_session):
    db_session.add_all([
        AppLog(timestamp="2026-06-10 09:00:00", level="INFO", module="m", message="hi", location="x:1"),
        AppLog(timestamp="2026-06-10 11:00:00", level="ERROR", module="m", message="boom", location="x:2"),
    ])
    await db_session.commit()

    assert (await query_logs(db_session, "app_logs", since="2026-06-10T10:00:00.000Z"))["total"] == 1
    # level filter = minimum severity → WARNING+ excludes INFO
    assert (await query_logs(db_session, "app_logs", level="WARNING"))["total"] == 1


async def test_query_frontend_type_filter(db_session):
    db_session.add_all([
        FrontendLog(timestamp="2026-06-10 10:00:00", type="network", message="a", page="/x"),
        FrontendLog(timestamp="2026-06-10 10:00:01", type="error", message="b", page="/y"),
    ])
    await db_session.commit()
    assert (await query_logs(db_session, "frontend_logs", type_filter="error"))["total"] == 1


async def test_query_unknown_table_raises(db_session):
    with pytest.raises(ValueError):
        await query_logs(db_session, "nope")


# ---- cleanup_logs ----

async def test_cleanup_age(db_session):
    db_session.add_all([
        AuditLog(timestamp="2000-01-01 00:00:00", action="old"),
        AuditLog(timestamp="2999-01-01 00:00:00", action="fresh"),
    ])
    await db_session.commit()
    deleted = await cleanup_logs(db_session, max_age_days=7)
    assert deleted["audit_logs"] == 1
    remaining = await query_logs(db_session, "audit_logs")
    assert remaining["total"] == 1
    assert remaining["items"][0]["action"] == "fresh"


async def test_cleanup_rowcap(db_session):
    for i in range(5):
        db_session.add(AppLog(timestamp="2999-01-01 00:00:00", level="INFO", message=f"m{i}"))
    await db_session.commit()
    await cleanup_logs(db_session, max_age_days=3650, max_rows=2)
    assert (await query_logs(db_session, "app_logs"))["total"] == 2


# ---- LogWriter (async queue + consumer) ----

async def test_writer_enqueue_flushes_to_db(tmp_path):
    factory = await _make_factory(tmp_path)
    writer = LogWriter()
    writer.start(session_factory=factory, batch_max=10)
    try:
        writer.enqueue("request", {"method": "GET", "path": "/p", "status": 200, "duration_ms": 5})
        writer.enqueue("audit", {"action": "load_engine", "path": "/x", "method": "POST"})
        # let the consumer drain
        for _ in range(50):
            await asyncio.sleep(0.01)
            async with factory() as s:
                res = await query_logs(s, "request_logs")
                if res["total"] >= 1:
                    break
        async with factory() as s:
            assert (await query_logs(s, "request_logs"))["total"] == 1
            audit = await query_logs(s, "audit_logs")
            assert audit["total"] == 1
            assert audit["items"][0]["timestamp"]  # stamped by enqueue
    finally:
        await writer.stop()


async def test_enqueue_before_start_is_noop():
    writer = LogWriter()
    # must not raise even though never started
    writer.enqueue("request", {"method": "GET", "path": "/p", "status": 200, "duration_ms": 1})
    assert not writer.started


async def test_queue_full_increments_dropped(tmp_path):
    factory = await _make_factory(tmp_path)
    writer = LogWriter()
    # tiny queue; don't run the consumer-drain race — fill faster than drain
    writer.start(session_factory=factory, maxsize=2, batch_max=1)
    try:
        # synchronously slam put_nowait past capacity (bypass loop hop to force fill)
        for i in range(20):
            writer._put_nowait(("request", {"method": "GET", "path": f"/{i}", "status": 200, "duration_ms": 1, "timestamp": "2026-06-10 10:00:00"}))
        assert writer.dropped > 0
    finally:
        await writer.stop()
