import pytest
import pytest_asyncio
from sqlalchemy import text
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
    """FTS 查询能按内容命中(走 idx_mem_content_fts 表达式 GIN 索引)。"""
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


@pytest.mark.asyncio
async def test_fts_gin_index_exists(pg_engine):
    """建表后 memory_entries 上必须有表达式 GIN 索引。

    没有它,prefetch 每次都要全表扫 + 逐行现算 to_tsvector。
    """
    async with pg_engine.connect() as conn:
        row = (await conn.execute(text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'memory_entries' AND indexname = 'idx_mem_content_fts'"
        ))).first()

    assert row is not None, "idx_mem_content_fts 不存在 —— 全文检索会退回全表扫"
    indexdef = row[0]
    assert "USING gin" in indexdef, indexdef
    # 表达式必须与 pg_provider.py 查询侧逐字一致,'simple' 不能丢(丢了 planner 不认)。
    assert "to_tsvector('simple'" in indexdef, indexdef


@pytest.mark.asyncio
async def test_fts_query_uses_index(pg_engine):
    """EXPLAIN 证明 FTS 谓词确实能走上索引,而不只是索引存在。

    测试库里数据量小,planner 多半更偏爱 seq scan —— 先 `SET enable_seqscan = off`
    把它逼到索引路径上,能出 Bitmap Index Scan 就说明表达式两侧对得上。
    """
    async with pg_engine.connect() as conn:
        await conn.execute(text("SET enable_seqscan = off"))
        rows = (await conn.execute(text(
            "EXPLAIN SELECT id FROM memory_entries "
            "WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', 'concise')"
        ))).all()

    plan = "\n".join(r[0] for r in rows)
    assert "Bitmap Index Scan" in plan or "idx_mem_content_fts" in plan, plan
