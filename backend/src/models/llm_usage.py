from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Index

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    instance_id = Column(BigInteger, nullable=True)
    api_key_id = Column(BigInteger, nullable=True)
    model = Column(String(128), nullable=False)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_llm_usage_model_created", "model", "created_at"),
        Index("idx_llm_usage_instance_created", "instance_id", "created_at"),
        Index("idx_llm_usage_created", "created_at"),
    )
