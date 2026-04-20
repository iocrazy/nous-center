"""Task 8 — record_llm_usage accepts optional agent_id.

`record_llm_usage` lives in `src.services.usage_service` and creates its
own session via `create_session_factory()`. These tests monkeypatch that
factory to point at a temp SQLite engine, then verify the row written.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.llm_usage import LLMUsage
from src.services import usage_service
from src.services.usage_service import record_llm_usage


async def _setup_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    return engine, sf


async def test_record_llm_usage_writes_agent_id(tmp_path, monkeypatch):
    engine, sf = await _setup_engine(tmp_path)
    monkeypatch.setattr(usage_service, "create_session_factory", lambda: sf)
    try:
        await record_llm_usage(
            model="qwen3.5",
            prompt_tokens=10,
            completion_tokens=20,
            agent_id="tutor",
        )
        async with sf() as session:
            row = (await session.execute(select(LLMUsage))).scalar_one()
            assert row.agent_id == "tutor"
    finally:
        await engine.dispose()


async def test_record_llm_usage_agent_id_optional(tmp_path, monkeypatch):
    """Omitting agent_id should write NULL."""
    engine, sf = await _setup_engine(tmp_path)
    monkeypatch.setattr(usage_service, "create_session_factory", lambda: sf)
    try:
        await record_llm_usage(
            model="qwen3.5",
            prompt_tokens=10,
            completion_tokens=20,
        )
        async with sf() as session:
            row = (await session.execute(select(LLMUsage))).scalar_one()
            assert row.agent_id is None
    finally:
        await engine.dispose()
