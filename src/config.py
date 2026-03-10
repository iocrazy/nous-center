from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://mindcenter:mindcenter@localhost:5432/mindcenter"

    NAS_MODELS_PATH: str = "/mnt/nas/models"
    NAS_OUTPUTS_PATH: str = "/mnt/nas/outputs"
    LOCAL_MODELS_PATH: str = "/media/heygo/Program/models"

    COSYVOICE_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/CosyVoice"
    INDEXTTS_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/index-tts"

    VLLM_BASE_URL: str = "http://localhost:8100"

    GPU_IMAGE: int = 0
    GPU_TTS: int = 1
    GPU_VIDEO: str = "0,1"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_model_configs(path: str = "configs/models.yaml") -> dict:
    with open(Path(path)) as f:
        data = yaml.safe_load(f)
    return data["models"]
