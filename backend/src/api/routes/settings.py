"""Settings management endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps_admin import require_admin
from src.config import get_settings, save_settings

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    local_models_path: str
    cosyvoice_repo_path: str
    indextts_repo_path: str
    gpu_image: int
    gpu_tts: int
    redis_url: str
    api_base_url: str = "http://localhost:8000"


class SettingsUpdate(BaseModel):
    local_models_path: str | None = None
    cosyvoice_repo_path: str | None = None
    indextts_repo_path: str | None = None
    gpu_image: int | None = None
    gpu_tts: int | None = None
    redis_url: str | None = None


@router.get("", response_model=SettingsResponse)
async def get_current_settings():
    """Return current configuration."""
    s = get_settings()
    return SettingsResponse(
        local_models_path=s.LOCAL_MODELS_PATH,
        cosyvoice_repo_path=s.COSYVOICE_REPO_PATH,
        indextts_repo_path=s.INDEXTTS_REPO_PATH,
        gpu_image=s.GPU_IMAGE,
        gpu_tts=s.GPU_TTS,
        redis_url=s.REDIS_URL,
    )


@router.put("", response_model=SettingsResponse, dependencies=[Depends(require_admin)])
async def update_settings(req: SettingsUpdate):
    """Update configuration and persist to settings.yaml."""
    updates = req.model_dump(exclude_none=True)

    # Map API field names to Settings attribute names
    field_map = {
        "local_models_path": "LOCAL_MODELS_PATH",
        "cosyvoice_repo_path": "COSYVOICE_REPO_PATH",
        "indextts_repo_path": "INDEXTTS_REPO_PATH",
        "gpu_image": "GPU_IMAGE",
        "gpu_tts": "GPU_TTS",
        "redis_url": "REDIS_URL",
    }

    settings_updates = {field_map[k]: v for k, v in updates.items() if k in field_map}
    save_settings(settings_updates)

    # Return updated settings
    s = get_settings()
    return SettingsResponse(
        local_models_path=s.LOCAL_MODELS_PATH,
        cosyvoice_repo_path=s.COSYVOICE_REPO_PATH,
        indextts_repo_path=s.INDEXTTS_REPO_PATH,
        gpu_image=s.GPU_IMAGE,
        gpu_tts=s.GPU_TTS,
        redis_url=s.REDIS_URL,
    )
