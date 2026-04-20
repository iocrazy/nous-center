"""Schema tests — agent_id column on response_sessions and llm_usage.

DB layer is async-only (`create_async_engine`), so these tests spin up an
in-memory SQLite engine via aiosqlite and inspect the synchronous metadata
within `run_sync`.
"""

import asyncio

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from src.models.database import Base
from src.models import response_session, llm_usage  # noqa: F401 ensure registered


def _inspect_columns(sync_conn, table: str):
    return [c["name"] for c in inspect(sync_conn).get_columns(table)]


async def _columns_for(table: str):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            return await conn.run_sync(lambda sc: _inspect_columns(sc, table))
    finally:
        await engine.dispose()


def test_response_session_has_agent_id_column():
    cols = asyncio.run(_columns_for("response_sessions"))
    assert "agent_id" in cols


def test_llm_usage_has_agent_id_column():
    cols = asyncio.run(_columns_for("llm_usage"))
    assert "agent_id" in cols
