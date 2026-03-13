from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, JSON, String

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class VoicePreset(Base):
    __tablename__ = "voice_presets"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
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

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    name = Column(String(100), nullable=False, unique=True)
    presets = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
