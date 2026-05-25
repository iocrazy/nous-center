# backend/src/services/component_scanner.py
"""component_scanner — enumerate unet/clip/vae/lora files by role for PR-4 loader nodes.

Modeled on lora_scanner.py: glob role-dirs declared in model_paths.yaml, detect
quant_type per file, cache module-level, expose invalidate. base_path from
settings.LOCAL_MODELS_PATH.

Spec §4.6.
"""
from __future__ import annotations

import glob as _glob
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROLE_DIRS = ("unet", "clip", "vae", "loras")

# HF-layout 分片:name-00001-of-00002.safetensors。同目录同 base 的分片是「一个模型拆成
# 多文件」,折叠成一个条目(否则下拉把每片当一个可选模型,误导;选单片语义也错)。
_SHARD_RE = re.compile(r"^(?P<base>.+)-\d{5}-of-\d{5}\.safetensors$")


def _collapse_shards(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同目录的多分片 → 一个条目(abs_path=首片,repo 推导兜回整模型;size=各片之和)。
    单文件原样保留。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    singles: list[dict[str, Any]] = []
    for e in entries:
        m = _SHARD_RE.match(e["filename"])
        if not m:
            singles.append(e)
            continue
        key = (str(Path(e["abs_path"]).parent), m.group("base"))
        groups.setdefault(key, []).append(e)
    collapsed: list[dict[str, Any]] = []
    for (_d, base), shards in groups.items():
        shards.sort(key=lambda e: e["abs_path"])
        first = shards[0]
        collapsed.append({
            "filename": f"{base}.safetensors",  # 去 -0000N-of-000MM 后缀
            "abs_path": first["abs_path"],       # 首片(loader 据此向上找 model_index)
            "size_mb": round(sum(s["size_mb"] for s in shards), 1),
            "quant_type": first["quant_type"],
            "mtime": first["mtime"],
            "shards": len(shards),
        })
    return singles + collapsed

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "model_paths.yaml"


def load_model_paths_config() -> dict[str, list[str]]:
    """Load role → glob-patterns from model_paths.yaml. Fail-soft to empty
    pattern lists if the file is missing OR malformed (keeps the scanner from
    crashing the app on a fresh checkout or a hand-edited broken config)."""
    if not _CONFIG_PATH.exists():
        logger.warning("model_paths.yaml not found at %s; component index will be empty", _CONFIG_PATH)
        return {role: [] for role in ROLE_DIRS}
    try:
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error("model_paths.yaml malformed (%s); component index will be empty", e)
        return {role: [] for role in ROLE_DIRS}
    roles = data.get("roles", {})
    return {role: list(roles.get(role, [])) for role in ROLE_DIRS}


def _base_path() -> Path:
    """LOCAL_MODELS_PATH from settings. Wrapped so tests can monkeypatch."""
    from src.config import get_settings
    return Path(get_settings().LOCAL_MODELS_PATH)


def _detect_quant_type(path: Path) -> str:
    """Filename-substring + extension based quant type. Mirrors quant_loaders matchers."""
    name = path.name.lower()
    if name.endswith(".gguf"):
        return "gguf"
    if "nvfp4mixed" in name:
        return "nvfp4mixed"
    if "mxfp8mixed" in name:
        return "mxfp8mixed"
    if "fp8mixed" in name:
        return "fp8mixed"
    if "fp16" in name or "float16" in name:
        return "fp16"
    return "bf16"


_cache: dict[str, list[dict[str, Any]]] | None = None


def scan_components(role: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return list of available files for a role. Cached module-level."""
    if role not in ROLE_DIRS:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLE_DIRS}")
    global _cache
    if _cache is None or force_refresh:
        _cache = _scan_all()
    return _cache.get(role, [])


def _scan_all() -> dict[str, list[dict[str, Any]]]:
    """Glob every role's patterns under base_path; build the full index."""
    base = _base_path()
    cfg = load_model_paths_config()
    index: dict[str, list[dict[str, Any]]] = {}
    for role, patterns in cfg.items():
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in _glob.glob(str(base / pattern), recursive=True):
                p = Path(match)
                if not p.is_file():
                    continue
                abs_path = str(p.resolve())
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                try:
                    stat = p.stat()
                    size_mb = round(stat.st_size / (1024 * 1024), 1)
                    mtime = stat.st_mtime
                except OSError:
                    size_mb, mtime = 0.0, 0.0
                entries.append({
                    "filename": p.name,
                    "abs_path": abs_path,
                    "size_mb": size_mb,
                    "quant_type": _detect_quant_type(p),
                    "mtime": mtime,
                })
        entries = _collapse_shards(entries)  # HF-layout 多分片 → 一个模型条目
        entries.sort(key=lambda e: e["filename"])
        index[role] = entries
    total = sum(len(v) for v in index.values())
    logger.info("component_scanner: indexed %d files across %d roles", total, len(index))
    return index


def get_component_index() -> dict[str, list[dict[str, Any]]]:
    """Full role → entries index. Populates cache if cold."""
    global _cache
    if _cache is None:
        _cache = _scan_all()
    return dict(_cache)


def invalidate_component_cache() -> None:
    """Drop the cache so the next scan re-globs."""
    global _cache
    _cache = None
