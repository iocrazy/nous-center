"""PostgreSQL-backed structured log store (replaces the standalone SQLite
``log_db``, spec 2026-06-10).

Writes go through a single async queue + one consumer task that batch-inserts
via the main DB engine. This preserves the three properties the old SQLite
design gave us:

1. **Writes never block the caller** — ``enqueue()`` is a sync, thread-safe,
   non-blocking hand-off (``loop.call_soon_threadsafe`` → ``put_nowait``).
2. **A write failure never propagates** — the consumer swallows DB errors and
   reports them to *stderr* (never back into ``logging``, which would recurse).
3. **Log writes don't fight the main connection pool** — one consumer holds at
   most one session at a time, and batches bursts into a single INSERT.

Reads/cleanup are plain async SQLAlchemy against the same engine.
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.log_entry import AppLog, AuditLog, FrontendLog, RequestLog

_CST = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(_CST).strftime("%Y-%m-%d %H:%M:%S")


# kind -> (model, allowed field names). enqueue() stamps `timestamp` itself.
_KINDS: dict[str, tuple[type, tuple[str, ...]]] = {
    "request": (RequestLog, ("method", "path", "status", "duration_ms", "ip", "user_agent")),
    "app": (AppLog, ("level", "module", "message", "location")),
    "frontend": (FrontendLog, ("type", "message", "page", "stack")),
    "audit": (AuditLog, ("action", "path", "method", "ip", "detail")),
}

# table name -> model, for the read/cleanup side (matches old log_db tables).
_TABLE_MODELS: dict[str, type] = {
    "request_logs": RequestLog,
    "app_logs": AppLog,
    "frontend_logs": FrontendLog,
    "audit_logs": AuditLog,
}

_SEARCH_COLS: dict[str, tuple[str, ...]] = {
    "request_logs": ("path", "method", "ip"),
    "app_logs": ("message", "module", "location"),
    "frontend_logs": ("message", "page", "type"),
    "audit_logs": ("action", "path", "detail"),
}

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


class LogWriter:
    """Async queue + single batch-insert consumer for log writes."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._session_factory = None
        self._batch_max = 200
        self.dropped = 0  # incremented when the queue is full (overload)

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, session_factory=None, maxsize: int = 10_000, batch_max: int = 200) -> None:
        """Start the consumer. Must be called from within the running loop."""
        if self.started:
            return
        if session_factory is None:
            from src.models.database import get_session_factory
            session_factory = get_session_factory()
        self._session_factory = session_factory
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._batch_max = batch_max
        self.dropped = 0
        self._task = self._loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def enqueue(self, kind: str, fields: dict) -> None:
        """Sync, thread-safe, non-blocking. Drops silently if not started / loop
        gone / queue full. NEVER raises into the caller."""
        if self._loop is None or self._queue is None:
            return
        fields = {**fields, "timestamp": _now()}
        try:
            self._loop.call_soon_threadsafe(self._put_nowait, (kind, fields))
        except RuntimeError:
            # loop already closed (shutdown) — drop
            pass

    def _put_nowait(self, item: tuple[str, dict]) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                await self._drain_and_flush()
                return
            batch = [first]
            # Drain whatever else is immediately available (burst coalescing).
            while len(batch) < self._batch_max:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._flush(batch)

    async def _drain_and_flush(self) -> None:
        if self._queue is None:
            return
        batch = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._flush(batch)

    async def _flush(self, batch: list[tuple[str, dict]]) -> None:
        rows = []
        for kind, fields in batch:
            spec = _KINDS.get(kind)
            if spec is None:
                continue
            model, allowed = spec
            kwargs = {k: fields[k] for k in allowed if k in fields}
            kwargs["timestamp"] = fields.get("timestamp") or _now()
            rows.append(model(**kwargs))
        if not rows:
            return
        try:
            async with self._session_factory() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as e:  # never let log writing crash the consumer
            print(f"[log_store] flush failed ({len(rows)} rows): {e}", file=sys.stderr)


# Process-wide singleton.
log_writer = LogWriter()


def enqueue(kind: str, fields: dict) -> None:
    log_writer.enqueue(kind, fields)


async def query_logs(
    session: AsyncSession,
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
    model = _TABLE_MODELS.get(table)
    if model is None:
        raise ValueError(f"Unknown table: {table}")

    conds = []
    if since:
        # Normalize ISO (2026-04-09T01:19:06.863Z) → "2026-04-09 01:19:06".
        normalized = since.replace("T", " ").replace("Z", "").split(".")[0]
        conds.append(model.timestamp >= normalized)
    if search:
        cols = [getattr(model, c) for c in _SEARCH_COLS.get(table, ())]
        if cols:
            conds.append(or_(*[c.like(f"%{search}%") for c in cols]))
    if level and table == "app_logs":
        min_level = _LEVEL_ORDER.get(level.upper(), 0)
        allowed = [k for k, v in _LEVEL_ORDER.items() if v >= min_level]
        conds.append(model.level.in_(allowed))
    if type_filter and table == "frontend_logs":
        conds.append(model.type == type_filter)
    if method and hasattr(model, "method"):
        conds.append(model.method == method.upper())
    if status and hasattr(model, "status"):
        if status.endswith("xx"):
            prefix = int(status[0])
            conds.append(model.status.between(prefix * 100, prefix * 100 + 99))
        else:
            conds.append(model.status == int(status))

    total = (await session.execute(select(func.count()).select_from(model).where(*conds))).scalar_one()
    rows = (
        await session.execute(
            select(model).where(*conds).order_by(model.id.desc()).limit(min(limit, 500)).offset(offset)
        )
    ).scalars().all()
    items = [{c.name: getattr(r, c.name) for c in model.__table__.columns} for r in rows]
    return {"total": total, "items": items}


async def cleanup_logs(
    session: AsyncSession, max_age_days: int = 7, max_rows: int = 100_000
) -> dict:
    cutoff = (datetime.now(_CST) - timedelta(days=max_age_days)).strftime("%Y-%m-%d %H:%M:%S")
    deleted: dict[str, int] = {}
    for table, model in _TABLE_MODELS.items():
        res = await session.execute(delete(model).where(model.timestamp < cutoff))
        deleted[table] = res.rowcount or 0
        # Row cap: find the id of the max_rows-th newest row, delete everything
        # strictly below it → keeps exactly the newest max_rows rows.
        threshold = (
            await session.execute(
                select(model.id).order_by(model.id.desc()).limit(1).offset(max(max_rows - 1, 0))
            )
        ).scalar_one_or_none()
        if threshold is not None:
            await session.execute(delete(model).where(model.id < threshold))
    await session.commit()
    return deleted
