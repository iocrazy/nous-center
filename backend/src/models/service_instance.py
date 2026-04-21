from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, JSON, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ServiceInstance(Base):
    __tablename__ = "service_instances"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    source_type = Column(String(20), nullable=False, default="preset")  # "preset", "workflow", or "model"
    source_id = Column(BigInteger, nullable=True, index=True)  # FK for preset/workflow
    source_name = Column(String(128), nullable=True)  # engine name for source_type="model"
    name = Column(String(100), nullable=False)
    type = Column(String(20), default="tts", nullable=False)  # tts, image, inference
    status = Column(String(20), default="active", nullable=False)
    # API-gateway additions (2026-04-21): marks this instance as a consumable
    # "service product". category groups instances for the external catalog;
    # meter_dim tells record_llm_usage + ResourcePack what to count.
    category = Column(String(20), nullable=True)  # "llm" | "tts" | "vl" | "app" | null
    meter_dim = Column(String(20), nullable=True)  # "tokens" | "chars" | "duration" | "calls" | null
    endpoint_path = Column(String(200), nullable=True)
    params_override = Column(JSON, default=dict)
    rate_limit_rpm = Column(Integer, nullable=True)
    rate_limit_tpm = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_service_instances_source", "source_type", "source_id"),
    )
