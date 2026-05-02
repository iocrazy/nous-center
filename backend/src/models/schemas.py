from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator


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
    emotion: str | None = None


class ImageUnderstandRequest(BaseModel):
    image_url: str
    question: str = "Describe this image in detail."
    model: str | None = None  # Override default VL model (settings.VL_MODEL)


# --- Responses ---

class TaskResponse(BaseModel):
    id: int
    task_type: str
    status: TaskStatus
    progress: int = 0
    result: dict | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}

    @field_serializer("id")
    def serialize_id(self, v: int) -> str:
        return str(v)


# --- Engine Management ---

class EngineInfo(BaseModel):
    name: str
    display_name: str
    type: str
    status: Literal["loaded", "unloaded", "loading", "failed"]
    gpu: int | list[int]
    vram_gb: float
    resident: bool
    local_path: str | None = None
    local_exists: bool = False
    status_detail: str | None = None
    # Remote metadata (from ModelScope / HuggingFace)
    organization: str | None = None
    model_size: str | None = None  # formatted: "494MB", "4.85GB"
    frameworks: list[str] | None = None
    libraries: list[str] | None = None
    license: str | None = None
    languages: list[str] | None = None
    tags: list[str] | None = None
    tensor_types: list[str] | None = None
    description: str | None = None
    has_metadata: bool = False
    auto_detected: bool = False
    # False when the model was discovered on disk but no adapter is wired
    # up to actually load it (e.g. ERNIE-Image: diffusers checkpoint with
    # no DiffusersImageAdapter implemented yet). UI uses this to disable
    # the load button instead of letting the user click into a confusing
    # "Unknown model" failure.
    has_adapter: bool = True
    loaded_gpu: int | None = None
    loaded_gpus: list[int] | None = None


class EngineLoadResponse(BaseModel):
    name: str
    status: Literal["loaded", "unloaded", "loading", "failed"]
    load_time_seconds: float | None = None


# --- Synchronous TTS (debug) ---

class SynthesizeRequest(BaseModel):
    engine: Literal[
        "cosyvoice2", "indextts2", "qwen3_tts_base",
        "qwen3_tts_customvoice", "qwen3_tts_voicedesign", "moss_tts",
    ]
    text: str
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None
    reference_text: str | None = None
    emotion: str | None = None
    cache: bool = True


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    engine: str
    rtf: float
    format: str = "wav"
    cached: bool = False


# --- SSE Streaming TTS ---

class StreamRequest(BaseModel):
    text: str
    engine: Literal[
        "cosyvoice2", "indextts2", "qwen3_tts_base",
        "qwen3_tts_customvoice", "qwen3_tts_voicedesign", "moss_tts",
    ] = "cosyvoice2"
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    reference_audio: str | None = None
    reference_text: str | None = None
    emotion: str | None = None
    cache: bool = True


# --- Voice Presets ---

class VoicePresetCreate(BaseModel):
    name: str
    engine: str
    params: dict = {}
    reference_audio_path: str | None = None
    reference_text: str | None = None
    tags: list[str] = []


class VoicePresetUpdate(BaseModel):
    name: str | None = None
    engine: str | None = None
    params: dict | None = None
    reference_audio_path: str | None = None
    reference_text: str | None = None
    tags: list[str] | None = None


class VoicePresetOut(BaseModel):
    id: int
    name: str
    engine: str
    params: dict
    reference_audio_path: str | None
    reference_text: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("id")
    def serialize_id(self, v: int) -> str:
        return str(v)


# --- Voice Preset Groups ---

class VoicePresetGroupCreate(BaseModel):
    name: str
    presets: list[str] = []  # list of preset names


class VoicePresetGroupOut(BaseModel):
    id: int
    name: str
    presets: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("id")
    def serialize_id(self, v: int) -> str:
        return str(v)


# --- Audio Upload ---

class AudioUploadResponse(BaseModel):
    id: str
    path: str
    duration_seconds: float | None = None


# --- Batch TTS (Round model) ---

class BatchRound(BaseModel):
    round_id: int
    voice_preset: str  # preset name
    text: str
    emotion: str | None = None


class BatchTTSRequest(BaseModel):
    rounds: list[BatchRound]


class BatchTTSResponse(BaseModel):
    batch_id: str
    total_rounds: int


class BatchRetryRequest(BaseModel):
    round_ids: list[int]


# --- Service Instances ---

class ServiceInstanceCreate(BaseModel):
    source_type: Literal["preset", "workflow", "model"] = "preset"
    source_id: int | None = None  # for preset/workflow
    source_name: str | None = None  # for model (engine name)
    name: str
    type: str = "tts"
    params_override: dict = {}

    @field_validator("source_id", mode="before")
    @classmethod
    def coerce_source_id(cls, v: int | str | None) -> int | None:
        if v is None:
            return None
        return int(v)


class ServiceInstanceUpdate(BaseModel):
    name: str | None = None
    params_override: dict | None = None
    rate_limit_rpm: int | None = None
    rate_limit_tpm: int | None = None


class ServiceInstanceOut(BaseModel):
    id: int
    source_type: str
    source_id: int | None = None
    source_name: str | None = None
    name: str
    type: str
    status: str
    endpoint_path: str | None
    params_override: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("id", "source_id")
    def serialize_ids(self, v: int | None) -> str | None:
        if v is None:
            return None
        return str(v)


class InstanceStatusUpdate(BaseModel):
    status: Literal["active", "inactive"]


# --- Instance API Keys ---

class InstanceApiKeyCreate(BaseModel):
    label: str


class InstanceApiKeyOut(BaseModel):
    id: int
    instance_id: int
    label: str
    key_prefix: str
    is_active: bool
    usage_calls: int
    usage_chars: int
    last_used_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("id", "instance_id")
    def serialize_ids(self, v: int) -> str:
        return str(v)


class InstanceApiKeyCreated(InstanceApiKeyOut):
    """Returned only on creation — includes the full key."""
    key: str


# --- Workflows ---

class WorkflowCreate(BaseModel):
    name: str
    description: str | None = None
    nodes: list = []
    edges: list = []
    is_template: bool = False


class WorkflowUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    nodes: list | None = None
    edges: list | None = None
    is_template: bool | None = None


class WorkflowOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    description: str | None
    nodes: list
    edges: list
    is_template: bool
    status: str
    auto_generated: bool = False
    generated_for_service_id: int | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("id")
    def serialize_id(self, v: int) -> str:
        return str(v)

    @field_serializer("generated_for_service_id")
    def serialize_back_link(self, v: int | None) -> str | None:
        return str(v) if v is not None else None


# --- v3 services / workflow publish (see routes/services.py + routes/workflow_publish.py) ---


class ExposedParam(BaseModel):
    """Shared shape for both inputs and outputs of a published service.

    v3 names (`key` / `input_name`) are preferred. The legacy aliases
    `api_name` / `param_key` are accepted for backward compat with rows
    backfilled from `workflow_apps`.
    """
    node_id: str
    key: str | None = None
    input_name: str | None = None
    label: str = ""
    type: str = "string"
    required: bool = True
    default: Any = None
    constraints: dict = {}
    # legacy aliases — accepted on input, ignored on output
    api_name: str | None = None
    param_key: str | None = None
