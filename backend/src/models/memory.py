"""Wave 1 memory tables (MemoryEntry + MemoryEmbedding)。

全局只有 PostgreSQL,索引直接声明在这里(含 FTS 的表达式 GIN 索引),
由 alembic 迁移落到库上。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, ForeignKey, Index, Integer,
    LargeBinary, String, Text, text,
)

from src.models.database import Base


class MemoryEntryModel(Base):
    __tablename__ = "memory_entries"

    # BigInteger + autoincrement → PG BIGSERIAL(全局只有 PG,无方言分叉)。
    id = Column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
    )
    # legacy rip:memory 按调用方 API key 切作用域。instance_id 降为 nullable 遗留列
    # (M:N key 无单一 instance);归属与查询走 api_key_id。
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    api_key_id = Column(BigInteger, nullable=True, index=True)
    category = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    context_key = Column(String(128), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        # 归属/查询走 api_key_id(prefetch 按 owner + context_key + 时间排序)。
        Index("idx_mem_key_created", "api_key_id", "created_at"),
        Index("idx_mem_key_ctx_cat", "api_key_id", "context_key", "category"),
        # 全文检索的表达式 GIN 索引。**表达式必须与查询侧逐字一致**
        # (pg_provider.py::prefetch 的 `to_tsvector('simple', content)`,含 'simple'):
        # planner 只在表达式完全匹配时才用得上这个索引,差一个配置名就退回全表扫 +
        # 逐行现算 tsvector。
        Index(
            "idx_mem_content_fts",
            text("to_tsvector('simple', content)"),
            postgresql_using="gin",
        ),
    )


class MemoryEmbeddingModel(Base):
    __tablename__ = "memory_embeddings"

    entry_id = Column(
        BigInteger,
        ForeignKey("memory_entries.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model = Column(String(64), nullable=False)
    dim = Column(Integer, nullable=False)
    vector = Column(LargeBinary, nullable=True)
