import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

import src.models.memory  # noqa: F401  register models on Base.metadata
from src.services.memory.pg_provider import PGMemoryProvider

from tests.test_memory_provider_abc import AbstractMemoryProviderTests


@pytest_asyncio.fixture
async def async_session_factory(pg_engine):
    # 表由 pg_engine fixture 建好;这个 provider 名字就叫 PG,以前却一直在 sqlite 上跑
    # ("against PG (SQLite in tests)"),JSONB / 全文索引这些 PG 专属路径从没被真正执行过。
    return async_sessionmaker(pg_engine, expire_on_commit=False)


class TestPGMemoryProviderContract(AbstractMemoryProviderTests):
    """Run the full AbstractMemoryProviderTests suite against a real PostgreSQL."""

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
        owner_key_id=1,
        entries=[
            {"category": "fact", "content": "user prefers concise replies", "context_key": None},
            {"category": "fact", "content": "user lives in Tokyo", "context_key": None},
        ],
    )
    results = await p.prefetch(owner_key_id=1, query="concise")
    assert len(results) == 1
    assert "concise" in results[0]["content"]
