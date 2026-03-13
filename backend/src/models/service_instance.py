from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, JSON, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ServiceInstance(Base):
    __tablename__ = "service_instances"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    source_type = Column(String(20), nullable=False, default="preset")  # "preset" or "workflow"
    source_id = Column(BigInteger, nullable=False, index=True)
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

    __table_args__ = (
        Index("ix_service_instances_source", "source_type", "source_id"),
    )
