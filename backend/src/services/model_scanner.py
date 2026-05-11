"""Auto-scan models directory to detect model type and config."""
import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_settings, load_model_configs

logger = logging.getLogger(__name__)


# V1' P0 reorganized image/ into ComfyUI-style subdirs. `diffusers/` holds
# full-layout model dirs (depth 3); the others hold single-file components
# (.safetensors / .gguf) that are NOT auto-detectable models on their own
# and should be enumerated by V1' Lane C component-node executors, not here.
_IMAGE_MODEL_SUBDIRS = {"diffusers"}
_IMAGE_COMPONENT_SUBDIRS = {"diffusion_models", "text_encoders", "vae"}


def _iter_candidate_model_dirs(type_dir: Path):
    """Yield (model_dir, local_path) pairs for one type/ tree.

    LLM/TTS/VL stay depth-2 (`<type>/<model>`). Image is depth-3 under the
    `diffusers/` sub-bucket since V1' P0, and we explicitly skip the
    component sub-buckets so their single-file weights aren't surfaced as
    bogus "models" in /engines (the V1' Lane C component nodes will list
    those separately).
    """
    if type_dir.name == "image":
        for sub in sorted(type_dir.iterdir()):
            if not sub.is_dir() or sub.name not in _IMAGE_MODEL_SUBDIRS:
                continue
            for model_dir in sorted(sub.iterdir()):
                if model_dir.is_dir():
                    yield model_dir, f"{type_dir.name}/{sub.name}/{model_dir.name}"
        return
    for model_dir in sorted(type_dir.iterdir()):
        if model_dir.is_dir():
            yield model_dir, f"{type_dir.name}/{model_dir.name}"


def scan_models() -> dict[str, dict[str, Any]]:
    """Scan LOCAL_MODELS_PATH and merge with models.yaml configs.

    Auto-detects:
    - LLM: has config.json with "model_type" field
    - Image (diffusers): has model_index.json (under `image/diffusers/<X>/`)
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

    for type_dir in sorted(base.iterdir()):
        if not type_dir.is_dir():
            continue
        for model_dir, local_path in _iter_candidate_model_dirs(type_dir):
            yaml_key = _find_yaml_key(yaml_configs, local_path)
            if yaml_key:
                continue
            detected = _detect_model(model_dir, local_path)
            if detected:
                key = _make_key(model_dir.name)
                result[key] = detected
                logger.info(
                    "Auto-detected model: %s (%s) at %s",
                    key, detected["type"], local_path,
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


_VLLM_ADAPTER = "src.services.inference.llm_vllm.VLLMAdapter"


def _detect_model(model_dir: Path, local_path: str) -> dict[str, Any] | None:
    """Auto-detect model type from directory contents.

    LLM and VL detections fill `adapter` so the registry can synthesize a
    ModelSpec on demand and the load button actually works. Image/video
    intentionally do NOT fill `adapter` — there's no diffusers adapter
    implemented yet, so leaving it blank lets the UI render a "未注册"
    badge and disable the load button instead of letting the user click
    into a confusing "Unknown model" failure.
    """

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
                    "adapter": _VLLM_ADAPTER,
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
                # NB: no `adapter` — diffusers adapter is unimplemented.
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
