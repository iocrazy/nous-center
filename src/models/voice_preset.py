import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, JSON, DateTime, TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from src.models.database import Base


class UUIDType(TypeDecorator):
    """Platform-independent UUID type. Uses PostgreSQL UUID when available,
    otherwise stores as CHAR(32)."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pgUUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
        else:
            return value.hex if isinstance(value, uuid.UUID) else uuid.UUID(value).hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(value)
        return value


class VoicePreset(Base):
    __tablename__ = "voice_presets"

    id = Column(UUIDType(length=32), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    engine = Column(String(50), nullable=False)
    params = Column(JSON, default=dict)
    reference_audio_path = Column(String(500), nullable=True)
    reference_text = Column(String(1000), nullable=True)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class VoicePresetGroup(Base):
    __tablename__ = "voice_preset_groups"

    id = Column(UUIDType(length=32), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    presets = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
