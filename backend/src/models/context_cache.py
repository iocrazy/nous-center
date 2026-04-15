"""Context Cache table — common_prefix mode metadata.

The actual prefix KV cache lives inside vLLM's GPU memory; this table tracks
lifecycle (TTL, hits, ownership) so clients can manage cache explicitly via the
/v1/context/* endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, Column, DateTime, ForeignKey,
    Index, Integer, String,
)
from sqlalchemy.dialects.postgresql import JSONB

from src.models.database import Base

# JSONB on PostgreSQL, plain JSON on SQLite (test fixture uses sqlite+aiosqlite)
JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def _expires_at_default():
    return datetime.now(timezone.utc) + timedelta(seconds=86400)


class ContextCache(Base):
    __tablename__ = "context_caches"

    id = Column(String(64), primary_key=True)
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # api_key_id is audit-only (creator); kept nullable so deleting a key
    # doesn't cascade-delete still-useful caches.
    api_key_id = Column(BigInteger, nullable=True)
    model = Column(String(128), nullable=False)
    mode = Column(String(32), nullable=False, default="common_prefix")
    messages_json = Column(JsonColumn, nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    ttl_seconds = Column(Integer, nullable=False, default=86400)
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_expires_at_default,
    )
    hit_count = Column(Integer, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "ttl_seconds >= 60 AND ttl_seconds <= 604800",
            name="context_cache_ttl_range",
        ),
        Index("ix_context_caches_expires_at", "expires_at"),
        Index("ix_context_caches_instance_expires", "instance_id", "expires_at"),
    )
