import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.models.memory  # noqa: F401  register models on Base.metadata
from src.models.database import Base
from src.services.memory.pg_provider import PGMemoryProvider

from tests.test_memory_provider_abc import AbstractMemoryProviderTests


@pytest_asyncio.fixture
async def async_session_factory(tmp_path):
    db_path = tmp_path / "pg_memory_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestPGMemoryProviderContract(AbstractMemoryProviderTests):
    """Run the full AbstractMemoryProviderTests suite against PG (SQLite in tests)."""

    @pytest_asyncio.fixture
    async def provider(self, async_session_factory):
        p = PGMemoryProvider(session_factory=async_session_factory)
        await p.initialize()
        yield p
        await p.shutdown()


@pytest.mark.asyncio
async def test_fts_index_hit(async_session_factory):
    """FTS GIN index should allow text search (PG only; SQLite fallback uses LIKE)."""
    p = PGMemoryProvider(session_factory=async_session_factory)
    await p.initialize()
    await p.add_entries(
        instance_id=1, api_key_id=None,
        entries=[
            {"category": "fact", "content": "user prefers concise replies", "context_key": None},
            {"category": "fact", "content": "user lives in Tokyo", "context_key": None},
        ],
    )
    results = await p.prefetch(instance_id=1, query="concise")
    assert len(results) == 1
    assert "concise" in results[0]["content"]
