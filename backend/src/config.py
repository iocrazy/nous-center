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
    # Directories scanned by lora_scanner (comma-separated).
    # Default includes ComfyUI loras dir so existing weights are reachable
    # without copy / symlink. Image specs in models.yaml don't need to list
    # individual LoRAs — registry injects them automatically into
    # spec.params['lora_paths'].
    LORA_PATHS: str = "/media/heygo/Program/models/comfyui/models/loras"

    COSYVOICE_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/CosyVoice"
    INDEXTTS_REPO_PATH: str = "/media/heygo/Program/projects-code/github-repos/index-tts"

    VLLM_BASE_URL: str = "http://localhost:8100"
    VL_MODEL: str = "Qwen2.5-VL-7B-Instruct"  # default model for /api/v1/understand

    GPU_IMAGE: int = 0
    GPU_TTS: int = 1
    GPU_VIDEO: str = "0,1"

    CACHE_TTL_SECONDS: int = 3600  # TTS cache TTL (1 hour)

    ADMIN_TOKEN: str = ""  # Set to require auth for management API (CLI/curl bearer token)
    # Browser admin login: when ADMIN_PASSWORD is set, /api/* and /ws/* require
    # a valid session cookie obtained via POST /sys/admin/login. Empty disables
    # the gate (dev mode). ADMIN_SESSION_SECRET signs the cookie HMAC.
    ADMIN_PASSWORD: str = ""
    ADMIN_SESSION_SECRET: str = ""
    ADMIN_SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 30  # 30 days
    # WebAuthn / Passkey settings.
    # ADMIN_PASSKEY_RP_ID is the host the browser sends — must EXACTLY match
    # the domain the page is loaded from (no scheme, no port). Examples:
    #   prod cloudflare:  api.iocrazy.com
    #   localhost dev:    localhost   (works without https for localhost only)
    # Multiple origins (dev + prod) are supported via a comma-separated list
    # in ADMIN_PASSKEY_RP_ORIGINS — every value must be `scheme://host[:port]`.
    ADMIN_PASSKEY_RP_ID: str = "localhost"
    ADMIN_PASSKEY_RP_NAME: str = "nous-center"
    ADMIN_PASSKEY_RP_ORIGINS: str = "http://localhost:9999,http://localhost:8000"

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
            # v2: `paths.main` is the canonical single-component path. Older
            # yaml using `path:` is not in the repo anymore (migrated by
            # PR-0 cutover) but we still gracefully read it as a fallback.
            paths = entry.get("paths") or {}
            # Single-component models (LLM/TTS) live under paths.main. Image
            # models are 3-component (transformer + text_encoder + vae) and
            # have no `main` — use the transformer's parent dir as the
            # canonical local_path so engines.py / scan_local_models can
            # match the entry against the on-disk directory.
            local_path = paths.get("main") or entry.get("path", "")
            if not local_path and paths.get("transformer"):
                from pathlib import Path as _P
                local_path = str(_P(paths["transformer"]).parent)
            result[model_id] = {
                "name": model_id,
                "type": entry.get("type", ""),
                # `gpu` stays None when unset so the GPU detector can auto-pick
                # a non-display card instead of defaulting to cuda:0.
                "gpu": entry.get("gpu"),
                "vram_gb": round(entry.get("vram_mb", 0) / 1024, 1),
                "resident": entry.get("resident", False),
                "local_path": local_path,
                "paths": paths,
                "ttl_seconds": entry.get("ttl_seconds", 300),
                # Preserve adapter so engines.py can compute has_adapter
                # without re-reading the yaml. Auto-detected entries fill
                # this same field from model_scanner.
                "adapter": entry.get("adapter"),
            }
            if entry.get("params"):
                result[model_id]["params"] = entry["params"]
        return result

    # Old dict-based format: return as-is
    return models
