"""Files API storage model (Step 5)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, ForeignKey, Index, String, UniqueConstraint,
)

from src.models.database import Base


class File(Base):
    __tablename__ = "files"

    id = Column(String(64), primary_key=True)  # file-{token_urlsafe(12)}
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    purpose = Column(String(32), nullable=False)
    filename = Column(String(512), nullable=False)
    bytes = Column(BigInteger, nullable=False)
    mime_type = Column(String(128), nullable=False)
    sha256 = Column(String(64), nullable=False)
    storage_path = Column(String(1024), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "instance_id", "sha256", name="uq_files_instance_sha256"
        ),
        Index(
            "ix_files_instance_created", "instance_id", "created_at"
        ),
    )
