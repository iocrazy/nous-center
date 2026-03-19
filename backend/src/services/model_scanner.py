"""Auto-scan models directory to detect model type and config."""
import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_settings, load_model_configs

logger = logging.getLogger(__name__)


def scan_models() -> dict[str, dict[str, Any]]:
    """Scan LOCAL_MODELS_PATH and merge with models.yaml configs.

    Auto-detects:
    - LLM: has config.json with "model_type" field
    - Image (diffusers): has model_index.json
    - TTS: matched by models.yaml only (no auto-detect)

    Returns merged dict: models.yaml configs + auto-detected models.
    """
    settings = get_settings()
    base = Path(settings.LOCAL_MODELS_PATH)
    yaml_configs = load_model_configs()

    # Start with yaml configs
    result = dict(yaml_configs)

    if not base.exists():
        return result

    # Scan subdirectories (depth 2: type/model-name)
    for type_dir in sorted(base.iterdir()):
        if not type_dir.is_dir():
            continue
        for model_dir in sorted(type_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            local_path = f"{type_dir.name}/{model_dir.name}"

            # Skip if already in yaml config (by matching local_path)
            yaml_key = _find_yaml_key(yaml_configs, local_path)
            if yaml_key:
                continue

            # Try auto-detect
            detected = _detect_model(model_dir, local_path)
            if detected:
                key = _make_key(model_dir.name)
                result[key] = detected
                logger.info(
                    "Auto-detected model: %s (%s) at %s",
                    key,
                    detected["type"],
                    local_path,
                )

    return result


def _find_yaml_key(configs: dict, local_path: str) -> str | None:
    """Find yaml config key that matches this local_path."""
    for key, cfg in configs.items():
        if cfg.get("local_path") == local_path:
            return key
    return None


def _make_key(dir_name: str) -> str:
    """Convert directory name to a config key."""
    return dir_name.lower().replace("-", "_").replace(".", "_")


def _detect_model(model_dir: Path, local_path: str) -> dict[str, Any] | None:
    """Auto-detect model type from directory contents."""

    # Check for HuggingFace LLM (config.json with model_type)
    config_json = model_dir / "config.json"
    if config_json.exists():
        try:
            with open(config_json) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            if model_type:
                # It's an LLM or VL model
                architectures = cfg.get("architectures", [])
                is_vl = any(
                    "VL" in a or "Vision" in a or "visual" in a.lower()
                    for a in architectures
                )

                return {
                    "name": model_dir.name,
                    "type": "understand" if is_vl else "llm",
                    "engine": "vllm",
                    "gpu": 0,
                    "vram_gb": _estimate_vram(model_dir),
                    "resident": False,
                    "local_path": local_path,
                    "auto_detected": True,
                }
        except (json.JSONDecodeError, OSError):
            pass

    # Check for diffusers model (model_index.json)
    model_index = model_dir / "model_index.json"
    if model_index.exists():
        try:
            with open(model_index) as f:
                idx = json.load(f)
            class_name = idx.get("_class_name", "")

            # Determine if image or video
            is_video = "video" in class_name.lower() or "wan" in model_dir.name.lower()

            return {
                "name": model_dir.name,
                "type": "video" if is_video else "image",
                "gpu": 0,
                "vram_gb": _estimate_vram(model_dir),
                "resident": False,
                "local_path": local_path,
                "auto_detected": True,
            }
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _estimate_vram(model_dir: Path) -> float:
    """Estimate VRAM from model file sizes (safetensors/bin/pt)."""
    total = 0
    for ext in ("*.safetensors", "*.bin", "*.pt", "*.onnx"):
        for f in model_dir.rglob(ext):
            total += f.stat().st_size

    gb = round(total / (1024**3), 1)
    # VRAM ~ 1.2x model file size (overhead for activations)
    return round(gb * 1.2, 1) if gb > 0 else 0
