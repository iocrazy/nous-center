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
    # legacy rip PR-5b:文件作用域从「绑服务 instance」改成「绑调用方 API key」(M:N 无单一 instance,
    # 谁上传谁拥有)。dedup 键 (api_key_id, sha256)。旧 instance_id 列经迁移降为 nullable 孤儿(开发期空表)。
    api_key_id = Column(
        BigInteger,
        ForeignKey("instance_api_keys.id", ondelete="CASCADE"),
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
            "api_key_id", "sha256", name="uq_files_apikey_sha256"
        ),
        Index(
            "ix_files_apikey_created", "api_key_id", "created_at"
        ),
    )
