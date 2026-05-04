"""LoRA discovery: walks Settings.LORA_PATHS and yields one entry per
.safetensors file. Subdirectories are walked recursively so ComfyUI's
bucketed layout (0_official/, 1_sdxl/, 5_sd1.5/, ...) surfaces every file.

Output shape (stable, consumed by /api/v1/loras + the registry injector):

    {
      "name":   <basename without .safetensors, subdir-prefixed when ambiguous>,
      "path":   <absolute path on disk>,
      "size_bytes": int,
      "subdir": <relative subdir under the search root, '' for root-level>,
    }

Names collide across buckets in real ComfyUI installs (people drop the same
LoRA into multiple subdirs). When that happens we keep the first by sort
order and prefix subsequent duplicates with "<subdir>/".
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)


_SUFFIXES = (".safetensors",)


def _search_dirs() -> list[Path]:
    raw = get_settings().LORA_PATHS
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def scan_loras() -> list[dict]:
    """Walk every configured LORA_PATHS directory; return a sorted list."""
    out: list[dict] = []
    seen_names: set[str] = set()

    for root in _search_dirs():
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in _SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            rel = path.relative_to(root)
            subdir = str(rel.parent) if rel.parent != Path(".") else ""
            stem = path.stem
            name = stem if stem not in seen_names else f"{subdir}/{stem}"
            if name in seen_names:
                # Even namespaced collision (extremely rare). Skip to keep
                # the (name → path) map injective downstream.
                logger.warning("LoRA name collision skipped: %s", path)
                continue
            seen_names.add(name)
            out.append({
                "name": name,
                "path": str(path),
                "size_bytes": size,
                "subdir": subdir,
            })

    out.sort(key=lambda e: e["name"].lower())
    return out


def get_lora_paths() -> dict[str, str]:
    """Convenience for the registry injector: name → absolute path."""
    return {entry["name"]: entry["path"] for entry in scan_loras()}
