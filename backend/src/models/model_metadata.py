from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, default=snowflake_id)
    engine_key: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    # Repo identifiers
    modelscope_id: Mapped[str | None] = mapped_column(String(200))
    hf_id: Mapped[str | None] = mapped_column(String(200))
    # Metadata from remote API
    organization: Mapped[str | None] = mapped_column(String(100))
    model_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    frameworks: Mapped[list | None] = mapped_column(JSON)
    libraries: Mapped[list | None] = mapped_column(JSON)
    license: Mapped[str | None] = mapped_column(String(100))
    languages: Mapped[list | None] = mapped_column(JSON)
    tags: Mapped[list | None] = mapped_column(JSON)
    tensor_types: Mapped[list | None] = mapped_column(JSON)
    description: Mapped[str | None] = mapped_column(Text)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
