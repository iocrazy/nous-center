from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, JSON, String

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ServiceInstance(Base):
    __tablename__ = "service_instances"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    preset_id = Column(
        BigInteger,
        ForeignKey("voice_presets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(100), nullable=False)
    type = Column(String(20), default="tts", nullable=False)  # tts, image, inference
    status = Column(String(20), default="active", nullable=False)
    endpoint_path = Column(String(200), nullable=True)
    params_override = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
