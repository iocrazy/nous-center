"""Responses API storage — event-sourcing (sessions + per-turn rows).

`previous_response_id` in the public API maps to `ResponseTurn.id` ("resp-xxx").
Internally, turns are grouped under a `ResponseSession` (id="session-xxx").
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, Column, DateTime, ForeignKey,
    Index, Integer, LargeBinary, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from src.models.database import Base

JsonColumn = JSON().with_variant(JSONB(), "postgresql")


def _expires_at_default():
    return datetime.now(timezone.utc) + timedelta(seconds=72 * 3600)


class ResponseSession(Base):
    __tablename__ = "response_sessions"

    id = Column(String(64), primary_key=True)  # session-{token_urlsafe(12)}
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    api_key_id = Column(BigInteger, nullable=True)
    model = Column(String(128), nullable=False)
    context_cache_id = Column(String(64), nullable=True)
    total_input_tokens = Column(BigInteger, nullable=False, default=0)
    total_output_tokens = Column(BigInteger, nullable=False, default=0)
    expire_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=_expires_at_default,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        # Keep CHECK simple; 7-day upper bound enforced at API layer to avoid
        # PG `interval` syntax breaking the SQLite test fixture.
        CheckConstraint(
            "expire_at > created_at",
            name="response_session_expire_at_check",
        ),
        Index(
            "ix_response_sessions_instance_created",
            "instance_id", "created_at",
        ),
    )


class ResponseTurn(Base):
    __tablename__ = "response_turns"

    id = Column(String(64), primary_key=True)  # resp-{token_urlsafe(12)}
    session_id = Column(
        String(64),
        ForeignKey("response_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_idx = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant
    content_compressed = Column(LargeBinary, nullable=False)
    usage_json = Column(JsonColumn, nullable=True)
    reasoning_json = Column(JsonColumn, nullable=True)
    instructions = Column(Text, nullable=True)
    text_format = Column(JsonColumn, nullable=True)
    status = Column(String(20), nullable=False, default="completed")
    incomplete_reason = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id", "turn_idx",
            name="uq_response_turn_session_idx",
        ),
        Index(
            "ix_response_turns_session_idx",
            "session_id", "turn_idx",
        ),
    )
