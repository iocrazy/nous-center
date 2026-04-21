from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

# All paths resolve relative to the backend/ directory (parent of src/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
SETTINGS_YAML_PATH = _BACKEND_DIR / "settings.yaml"


class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql+asyncpg://mindcenter:mindcenter@localhost:5432/mindcenter"

    NAS_MODELS_PATH: str = "/mnt/nas/models"
    NAS_OUTPUTS_PATH: str = "/mnt/nas/outputs"
    LOCAL_MODELS_PATH: str = "/media/heygo/Program/models/nous"

    COSYVOICE_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/CosyVoice"
    INDEXTTS_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/index-tts"

    VLLM_BASE_URL: str = "http://localhost:8100"
    VL_MODEL: str = "Qwen2.5-VL-7B-Instruct"  # default model for /api/v1/understand

    GPU_IMAGE: int = 0
    GPU_TTS: int = 1
    GPU_VIDEO: str = "0,1"

    CACHE_TTL_SECONDS: int = 3600  # TTS cache TTL (1 hour)

    ADMIN_TOKEN: str = ""  # Set to require auth for management API

    NOUS_CENTER_HOME: str = "~/.nous-center"

    NOUS_ENABLE_AGENT_INJECTION: bool = False  # feature flag for agent/skill system prompt injection

    model_config = {"env_file": ".env", "extra": "ignore"}


def _resolve_path(relative: str) -> Path:
    """Resolve a path relative to the backend/ directory."""
    return _BACKEND_DIR / relative


@lru_cache
def get_settings() -> Settings:
    overrides = _load_settings_yaml()
    return Settings(**overrides)


def _load_settings_yaml() -> dict:
    """Load overrides from settings.yaml if it exists."""
    if not SETTINGS_YAML_PATH.exists():
        return {}
    try:
        with open(SETTINGS_YAML_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


def save_settings(updates: dict) -> None:
    """Merge updates into settings.yaml and clear the cached Settings."""
    existing = _load_settings_yaml()
    existing.update(updates)

    with open(SETTINGS_YAML_PATH, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    get_settings.cache_clear()


def load_model_configs(path: str = "configs/models.yaml") -> dict:
    """Load model configs and return dict keyed by model id/name.

    Supports both old dict-based format and new list-based format.
    """
    resolved = _resolve_path(path)
    with open(resolved) as f:
        data = yaml.safe_load(f)
    models = data["models"]

    # New list-based format: convert to dict keyed by id
    if isinstance(models, list):
        result = {}
        for entry in models:
            model_id = entry["id"]
            result[model_id] = {
                "name": model_id,
                "type": entry.get("type", ""),
                # `gpu` stays None when unset so the GPU detector can auto-pick
                # a non-display card instead of defaulting to cuda:0.
                "gpu": entry.get("gpu"),
                "vram_gb": round(entry.get("vram_mb", 0) / 1024, 1),
                "resident": entry.get("resident", False),
                "local_path": entry.get("path", ""),
                "ttl_seconds": entry.get("ttl_seconds", 300),
            }
            if entry.get("params"):
                result[model_id]["params"] = entry["params"]
        return result

    # Old dict-based format: return as-is
    return models
