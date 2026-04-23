from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, default=snowflake_id)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    edges: Mapped[list] = mapped_column(JSON, default=list)
    is_template: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="draft")

    # v3 quick-provision: services created via the wizard generate a 1-3
    # node trivial workflow that backs them. We mark those with
    # auto_generated=True so the m08 workflow list can hide them by default
    # (filter `auto_generated=false`). generated_for_service_id is the back
    # link (ON DELETE SET NULL keeps the orphan workflow around for forensics).
    auto_generated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    generated_for_service_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
