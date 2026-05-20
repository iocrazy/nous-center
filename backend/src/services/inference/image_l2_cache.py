"""PR-6: L2 image_generate output cache (spec §3.3). Per-runner, in-memory,
LRU(50). Keyed by everything that determines the image; only deterministic
(seeded) runs participate. Stores a disk anchor (uuid/date/ext/meta) and
re-signs a fresh URL on each hit — never caches an expiring URL."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any


def image_l2_key(node, req) -> str:
    """Deterministic hash of all output-affecting inputs. `node` supplies the
    legacy model_key; `req` supplies prompt/sampling/components."""
    from src.services.inference.component_spec import to_component_key

    comps = getattr(req, "components", None)
    if comps:
        comp_part: list = []
        for kind in ("unet", "clip", "vae"):
            f, dev, dt, lset = to_component_key(comps[kind])
            comp_part.append([kind, f, dev, dt, sorted(list(lset))])
        model_part: dict[str, Any] = {
            "pipeline_class": getattr(req, "pipeline_class", None),
            "components": comp_part,
        }
    else:
        model_part = {
            "model_key": getattr(node, "model_key", None),
            "loras": sorted([(lo.name, float(lo.strength)) for lo in getattr(req, "loras", [])]),
        }
    payload = {
        "model": model_part,
        "prompt": req.prompt,
        "negative": req.negative_prompt,
        "steps": req.steps,
        "w": req.width,
        "h": req.height,
        "cfg": req.cfg_scale,
        "seed": req.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class ImageOutputCache:
    """LRU anchor cache. Entry: {image_uuid, date, ext, meta, width, height}."""

    def __init__(self, maxsize: int = 50) -> None:
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        self._max = maxsize

    def get(self, key: str) -> dict | None:
        entry = self._d.get(key)
        if entry is not None:
            self._d.move_to_end(key)
        return entry

    def put(self, key: str, entry: dict) -> None:
        self._d[key] = entry
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)


def serve_image_l2(entry: dict, ttl: int) -> dict | None:
    """Build a NodeResult outputs payload from a cache entry, re-signing a fresh
    URL. Returns None when the underlying PNG was reaped (caller treats as miss)."""
    from src.services.image_output_storage import resolve_path, sign_existing_image

    path = resolve_path(entry["date"], entry["image_uuid"], entry["ext"])
    if not path.exists():
        return None
    url, expires = sign_existing_image(entry["date"], entry["image_uuid"], entry["ext"], ttl_seconds=ttl)
    return {
        "image_url": url,
        "image_uuid": entry["image_uuid"],
        "image_expires": expires,
        "width": entry.get("width"),
        "height": entry.get("height"),
        "meta": entry.get("meta"),
        "media_type": f"image/{entry['ext']}",
        "cached": True,
    }
