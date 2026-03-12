from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class TTSUsage(Base):
    __tablename__ = "tts_usage"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    engine = Column(String(64), nullable=False)
    characters = Column(Integer, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    rtf = Column(Float, nullable=True)
    cached = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_tts_usage_created", "created_at"),
        Index("idx_tts_usage_engine", "engine"),
    )
