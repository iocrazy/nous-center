"""Wave 1 memory tables (MemoryEntry + MemoryEmbedding).

Dialect-agnostic declarations; FTS index is PG-only and added in raw SQL migration.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, ForeignKey, Index, Integer,
    LargeBinary, String, Text,
)

from src.models.database import Base


class MemoryEntryModel(Base):
    __tablename__ = "memory_entries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    api_key_id = Column(BigInteger, nullable=True)
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
        Index("idx_mem_inst_created", "instance_id", "created_at"),
        Index("idx_mem_inst_key_cat", "instance_id", "context_key", "category"),
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
