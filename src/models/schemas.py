import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Requests ---

class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    num_steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = None


class VideoGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=832, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=720)
    num_frames: int = Field(default=81, ge=1, le=161)
    seed: int | None = None


class TTSRequest(BaseModel):
    text: str
    engine: Literal[
        "cosyvoice2",
        "indextts2",
        "qwen3_tts_base",
        "qwen3_tts_customvoice",
        "qwen3_tts_voicedesign",
        "moss_tts",
    ] = "cosyvoice2"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None  # for voice cloning engines


class ImageUnderstandRequest(BaseModel):
    image_url: str
    question: str = "Describe this image in detail."


# --- Responses ---

class TaskResponse(BaseModel):
    id: uuid.UUID
    task_type: str
    status: TaskStatus
    progress: int = 0
    result: dict | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# --- Engine Management ---

class EngineInfo(BaseModel):
    name: str
    display_name: str
    type: str
    status: Literal["loaded", "unloaded"]
    gpu: int
    vram_gb: float
    resident: bool


class EngineLoadResponse(BaseModel):
    name: str
    status: Literal["loaded", "unloaded"]
    load_time_seconds: float | None = None
