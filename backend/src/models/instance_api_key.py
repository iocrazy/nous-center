from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String, Text

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class InstanceApiKey(Base):
    __tablename__ = "instance_api_keys"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    # API-gateway transition (2026-04-21): instance_id is now nullable.
    # Legacy 1:1 binding when set (pre-existing keys); new M:N binding when
    # null, with grants living in api_key_grants. verify_bearer_token
    # handles both paths for backward compat.
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    label = Column(String(100), nullable=False)
    key_hash = Column(String(200), nullable=False)
    key_prefix = Column(String(20), nullable=False)
    # m10: always-visible mode (Aliyun Bailian style). Stored alongside
    # key_hash so existing bcrypt verification still works for keys with no
    # plaintext (legacy or rotated). UI gates "view" on its presence.
    secret_plaintext = Column(String(200), nullable=True)
    note = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    usage_calls = Column(Integer, default=0, nullable=False)
    usage_chars = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
