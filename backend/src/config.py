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


# 运行时/每机覆盖(resident 常驻、gpu 指派、vram_budget 显存预算)。
# 数据加载统一(2026-06-16,用户拍「拆数据表」):从 gitignore 的 runtime_overrides.json
# 文件迁到 Postgres typed 表 model_runtime_overrides(与服务/key 同库),拆成正经列。
# 读取仍同步(走 runtime_override_store 的进程内缓存,启动时从 DB hydrate);写在异步
# API handler 里走 runtime_override_store.set_override。**叠加在 models.yaml 之上**(overlay 优先)。
# 旧 JSON 路径仅留作一次性迁移源(migrate_json_if_empty)。
_RUNTIME_OVERRIDES_REL = "configs/runtime_overrides.json"
# vram_budget:每模型显存预算({"mode":"auto|percent|absolute","value":N})。
_OVERRIDABLE_KEYS = ("resident", "gpu", "vram_budget")


def load_runtime_overrides() -> dict:
    """运行时覆盖快照(同步)。真相源 = DB(runtime_override_store 缓存,启动 hydrate)。
    局部 import 断 config↔store↔database↔config 循环。未 hydrate(早期/无 DB 的测试)→ 空 dict。"""
    from src.services import runtime_override_store  # noqa: PLC0415
    return runtime_override_store.get_overrides()


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


def recommend_vram_budget_gb(model_type: str, weights_gb: float) -> float:
    """每模型显存预算推荐值(GB)—— 权重 + 该模态典型激活/KV 余量(spec 2026-06-13)。
    embedding/tts 几乎不用 KV(单次前向)→ ×1.25;llm/vl 自回归解码要 KV → +6G 一档实用。
    其余(保守)×1.3。给 UI 显示「推荐」+ auto 落地参考。"""
    w = max(0.0, float(weights_gb or 0))
    t = (model_type or "").lower()
    if t in ("embedding", "tts"):
        rec = w * 1.25
    elif t in ("llm", "understand", "vl"):
        rec = w + 6.0
    else:
        rec = w * 1.3
    return round(max(1.0, rec), 1)


def resolve_vram_utilization(
    vram_budget: dict | None,
    gpu_total_gb: float,
    fallback: float | None,
    auto_util: float,
) -> float:
    """vram_budget({mode,value}) → vLLM gpu_memory_utilization(0–1)。
    优先级:显式 overlay vram_budget(percent/absolute)> models.yaml 的 fallback
    (gpu_memory_utilization)> auto 公式。mode=auto 或缺省 → 走 fallback/auto。
    absolute(GB)按目标卡真实总显存换算成比例;clamp 到 (0, 0.98]。"""
    if isinstance(vram_budget, dict):
        mode = str(vram_budget.get("mode") or "auto").lower()
        val = vram_budget.get("value")
        if mode == "percent" and isinstance(val, (int, float)) and val > 0:
            return max(0.01, min(0.98, float(val)))
        if mode == "absolute" and isinstance(val, (int, float)) and val > 0 and gpu_total_gb > 0:
            return max(0.01, min(0.98, float(val) / gpu_total_gb))
    if fallback:
        return max(0.01, min(0.98, float(fallback)))
    return auto_util


def _apply_runtime_overrides(cfgs: dict) -> None:
    """把 runtime_overrides.json 的 resident/gpu/vram_budget 叠加进 cfgs(原地改)。overlay 优先于 models.yaml。"""
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
