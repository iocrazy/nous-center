import json
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
    # 图像/产物签名 URL 有效期(秒)。服务层 API spec PR-4:输出交付 TTL 归服务层配置,
    # 不再是每个出图节点的 widget(用户:URL 有效期不该是节点的事,该是工作流 API 的功能)。
    IMAGE_URL_TTL_SECONDS: int = 3600

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
        # round4 #5(config):手改成 YAML 列表/标量时 safe_load 返回非 dict(truthy 不被
        # `or {}` 兜),get_settings 的 `Settings(**data)` 会 TypeError 崩。对齐
        # load_hardware_config 的 isinstance 守卫,非 mapping 降级空 dict。
        if not isinstance(data, dict):
            return {}
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


# 运行时/每机覆盖(resident 常驻、gpu 指派等)。存 gitignore 的 runtime_overrides.json,
# **叠加在 models.yaml 之上**(overlay 优先)。原因:models.yaml 是 git 跟踪的,UI 设的常驻
# 写进去就是未提交本地改动,任何 git checkout/pull/reset 都会冲掉(用户报告:标了常驻还是丢)。
# 把运行时状态分离到不跟踪的 overlay,既即时持久又不被 git 动。形如 {"<model_id>": {"resident": true}}。
_RUNTIME_OVERRIDES_REL = "configs/runtime_overrides.json"
_OVERRIDABLE_KEYS = ("resident", "gpu")


def load_runtime_overrides() -> dict:
    # Path(...) 包一层:_resolve_path 正常返回 Path,但测试会 monkeypatch 它返回 str
    # (test_image_model_integration),str 没有 .exists() —— 包一层两种都安全。
    p = Path(_resolve_path(_RUNTIME_OVERRIDES_REL))
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — 坏文件不该拖垮模型加载
        return {}


def set_runtime_override(model_id: str, key: str, value) -> None:
    """写一条运行时覆盖(如 resident),持久到 gitignore 的 overlay,不碰 git 跟踪的 models.yaml。"""
    if key not in _OVERRIDABLE_KEYS:
        raise ValueError(f"non-overridable key: {key}")
    p = Path(_resolve_path(_RUNTIME_OVERRIDES_REL))
    data = load_runtime_overrides()
    data.setdefault(model_id, {})[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_model_configs(path: str = "configs/models.yaml") -> dict:
    """Load model configs and return dict keyed by model id/name.

    Supports both old dict-based format and new list-based format.
    运行时覆盖(resident/gpu)叠加在最后,见 load_runtime_overrides。
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
            # V1' P2: optional `files{}` declares which on-disk files compose
            # this preset (transformer / text_encoder / vae). Lane C component
            # nodes consume it for dropdowns; the adapter still loads via paths.
            if entry.get("files"):
                result[model_id]["files"] = entry["files"]
        _apply_runtime_overrides(result)
        return result

    # Old dict-based format: return as-is
    _apply_runtime_overrides(models)
    return models


def _apply_runtime_overrides(cfgs: dict) -> None:
    """把 runtime_overrides.json 的 resident/gpu 叠加进 cfgs(原地改)。overlay 优先于 models.yaml。"""
    overrides = load_runtime_overrides()
    for mid, ov in overrides.items():
        if mid in cfgs and isinstance(ov, dict):
            for k in _OVERRIDABLE_KEYS:
                if k in ov:
                    cfgs[mid][k] = ov[k]


@lru_cache
def load_hardware_config(path: str = "configs/hardware.yaml") -> dict:
    """Load the manual GPU topology config (hardware.yaml).

    Returns a dict with a "groups" list. fail-soft: missing file, corrupt
    YAML, or missing "groups" key all return {"groups": []} so the
    GPUAllocator can degrade to detect-based single-card groups instead
    of crashing API server startup (spec §3.2, manual-only topology).
    """
    # path may be absolute (tests) or relative-to-backend (default).
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else _resolve_path(path)
    if not resolved.exists():
        return {"groups": []}
    try:
        with open(resolved) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {"groups": []}
    groups = data.get("groups")
    if not isinstance(groups, list):
        return {"groups": []}
    return {"groups": groups}
