# backend/src/services/component_scanner.py
"""component_scanner — enumerate unet/clip/vae/lora files by role for PR-4 loader nodes.

Modeled on lora_scanner.py: glob role-dirs declared in model_paths.yaml, detect
quant_type per file, cache module-level, expose invalidate. base_path from
settings.LOCAL_MODELS_PATH.

Spec §4.6.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROLE_DIRS = ("unet", "clip", "vae", "loras")

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "model_paths.yaml"


def load_model_paths_config() -> dict[str, list[str]]:
    """Load role → glob-patterns from model_paths.yaml. Fail-soft to empty
    pattern lists if the file is missing."""
    if not _CONFIG_PATH.exists():
        logger.warning("model_paths.yaml not found at %s; component index will be empty", _CONFIG_PATH)
        return {role: [] for role in ROLE_DIRS}
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    roles = data.get("roles", {})
    return {role: list(roles.get(role, [])) for role in ROLE_DIRS}
