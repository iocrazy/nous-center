from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class PresetApiKey(Base):
    __tablename__ = "preset_api_keys"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    preset_id = Column(
        BigInteger,
        ForeignKey("voice_presets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label = Column(String(100), nullable=False)
    key_hash = Column(String(200), nullable=False)
    key_prefix = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    usage_calls = Column(Integer, default=0, nullable=False)
    usage_chars = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
