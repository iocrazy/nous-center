from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class WorkflowApp(Base):
    __tablename__ = "workflow_apps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, default=snowflake_id)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    workflow_id: Mapped[int] = mapped_column(BigInteger)
    workflow_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    exposed_inputs: Mapped[list] = mapped_column(JSON, default=list)
    exposed_outputs: Mapped[list] = mapped_column(JSON, default=list)
    call_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
