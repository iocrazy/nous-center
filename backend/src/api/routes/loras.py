"""GET /api/v1/loras — list discovered LoRA weights for the image_generate
node's LoRA stack widget.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.response_cache import cached
from src.services.lora_scanner import scan_loras

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])


@router.get("")
@cached("loras", ttl=30)
async def list_loras():
    """Return discovered LoRA weights as a sorted list.

    Frontend uses these for the image_generate node's LoRA stack dropdown.
    Backend image adapter reads the same data via spec.params['lora_paths']
    (registry-injected); the route exists purely for the UI.
    """
    return [
        {
            "name": entry["name"],
            "size_mb": round(entry["size_bytes"] / (1024 * 1024), 1),
            "subdir": entry["subdir"],
        }
        for entry in scan_loras()
    ]
