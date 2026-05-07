"""LoRA discovery: walks Settings.LORA_PATHS and yields one entry per
.safetensors file. Subdirectories are walked recursively so ComfyUI's
bucketed layout (0_official/, 1_sdxl/, 5_sd1.5/, ...) surfaces every file.

Output shape (stable, consumed by /api/v1/loras + the registry injector):

    {
      "name":   <basename without .safetensors, subdir-prefixed when ambiguous>,
      "path":   <absolute path on disk>,
      "size_bytes": int,
      "subdir": <relative subdir under the search root, '' for root-level>,
      "arch":   one of 'flux2'|'flux1'|'sdxl'|'sd1.5'|'sd3'|'ernie'|'unknown'
    }

Names collide across buckets in real ComfyUI installs (people drop the same
LoRA into multiple subdirs). When that happens we keep the first by sort
order and prefix subsequent duplicates with "<subdir>/".

Architecture detection (V0.6 P4):
1. Try `modelspec.architecture` / `ss_base_model_version` from safetensors
   metadata (kohya / sai standards).
2. Fallback to key-prefix sniff:
   - lora_te1_/lora_te2_   → SDXL
   - lora_te_text_*         → SD1.5 (legacy kohya)
   - double_blocks/single_blocks → Flux1/2 (can't fully distinguish without dim sniff)
   - transformer_blocks (no input_blocks) → SD3 / diffusers PEFT format
3. Fallback to subdir name (1_sdxl, 5_sd1.5, etc).
4. Last resort: 'unknown'.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)


_SUFFIXES = (".safetensors",)

# Cheap in-process cache so /api/v1/engines pulls (~1 req/s when /models page
# is open) don't reopen 12+ safetensors every time. 5 min TTL is plenty for
# a single-admin scanner — re-scan happens on /scan endpoint or restart.
_SCAN_CACHE: dict = {"data": None, "ts": 0.0}
_SCAN_TTL_SECONDS = 300


def _arch_from_metadata(meta: dict | None) -> str | None:
    if not meta:
        return None
    arch_raw = (meta.get("modelspec.architecture") or "").lower()
    base = (meta.get("ss_base_model_version") or "").lower()
    if "flux-2" in arch_raw or "flux2" in arch_raw or "klein" in arch_raw:
        return "flux2"
    if "flux-1" in arch_raw or "flux1" in arch_raw or arch_raw == "flux":
        return "flux1"
    if "stable-diffusion-xl" in arch_raw or base in ("sdxl_base_v1", "sdxl"):
        return "sdxl"
    if "stable-diffusion-v1" in arch_raw or base in ("sd_v1", "sd1.5"):
        return "sd1.5"
    if "stable-diffusion-v3" in arch_raw or "sd3" in arch_raw:
        return "sd3"
    if "ernie" in arch_raw:
        return "ernie"
    return None


def _arch_from_keys(keys: list[str]) -> str | None:
    keyset = set(keys[:50])  # first 50 keys are class-level structural
    has_double_blocks = any("double_blocks" in k for k in keyset)
    has_single_blocks = any("single_blocks" in k for k in keyset)
    has_transformer_blocks = any("transformer_blocks" in k for k in keyset)
    has_te1 = any(k.startswith("lora_te1_") for k in keyset)
    has_te2 = any(k.startswith("lora_te2_") for k in keyset)
    has_te_legacy = any(k.startswith("lora_te_text_model") for k in keyset)
    has_input_blocks = any("input_blocks" in k for k in keyset)

    if has_double_blocks and has_single_blocks:
        # Flux family — we default to flux1 (vastly more public LoRAs).
        # flux2 LoRAs are rare and usually metadata-tagged.
        return "flux1"
    if has_transformer_blocks and not has_input_blocks:
        # diffusers PEFT format (SD3 / Flux2 / new arches). Default to sd3
        # as the most common — flux2 PEFT LoRAs typically tag metadata.
        return "sd3"
    if has_te1 and has_te2:
        return "sdxl"
    if has_te_legacy and has_input_blocks:
        return "sd1.5"
    return None


def _arch_from_subdir(subdir: str) -> str | None:
    s = subdir.lower()
    if "flux2" in s or "klein" in s:
        return "flux2"
    if "flux" in s:
        return "flux1"
    if "sdxl" in s:
        return "sdxl"
    if "sd1.5" in s or "sd15" in s:
        return "sd1.5"
    if "sd3" in s:
        return "sd3"
    if "ernie" in s:
        return "ernie"
    return None


def _detect_arch(path: Path, subdir: str) -> str:
    """3-tier detection: metadata → key prefix → subdir → 'unknown'."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
            arch = _arch_from_metadata(meta)
            if arch:
                return arch
            keys = list(f.keys())
            arch = _arch_from_keys(keys)
            if arch:
                return arch
    except Exception as e:
        logger.warning("LoRA arch sniff failed for %s: %s", path, e)

    arch = _arch_from_subdir(subdir)
    if arch:
        return arch
    return "unknown"


def _search_dirs() -> list[Path]:
    raw = get_settings().LORA_PATHS
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def scan_loras(*, force_refresh: bool = False) -> list[dict]:
    """Walk every configured LORA_PATHS directory; return a sorted list.

    Cached for 5 min. Cache is keyed on the raw LORA_PATHS string so tests
    that monkeypatch settings between cases don't see stale data. Pass
    `force_refresh=True` to bypass cache (used by POST /api/v1/engines/scan).
    """
    paths_key = get_settings().LORA_PATHS
    now = time.time()
    if (
        not force_refresh
        and _SCAN_CACHE["data"] is not None
        and _SCAN_CACHE.get("paths_key") == paths_key
        and now - _SCAN_CACHE["ts"] < _SCAN_TTL_SECONDS
    ):
        return _SCAN_CACHE["data"]  # type: ignore[return-value]

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
                logger.warning("LoRA name collision skipped: %s", path)
                continue
            seen_names.add(name)
            arch = _detect_arch(path, subdir)
            out.append({
                "name": name,
                "path": str(path),
                "size_bytes": size,
                "subdir": subdir,
                "arch": arch,
            })

    out.sort(key=lambda e: e["name"].lower())
    _SCAN_CACHE["data"] = out
    _SCAN_CACHE["ts"] = now
    _SCAN_CACHE["paths_key"] = paths_key
    return out


def count_loras_for_arches(accepts: list[str]) -> int:
    """Count LoRAs whose detected arch is in `accepts`. Empty list returns
    full count (model declares "accepts everything", e.g. legacy yaml entries)."""
    if not accepts:
        return len(scan_loras())
    accepts_set = set(accepts)
    return sum(1 for e in scan_loras() if e["arch"] in accepts_set)


def get_lora_paths() -> dict[str, str]:
    """Convenience for the registry injector: name → absolute path."""
    return {entry["name"]: entry["path"] for entry in scan_loras()}


def invalidate_cache() -> None:
    """Force the next scan_loras() to re-walk disk + re-sniff archs."""
    _SCAN_CACHE["data"] = None
    _SCAN_CACHE["ts"] = 0.0
