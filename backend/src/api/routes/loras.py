"""GET /api/v1/loras — list discovered LoRA weights for the image_generate
node's LoRA stack widget.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.response_cache import cached
from src.services.lora_scanner import scan_loras

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])


@router.get("")
@cached("loras", ttl=30)
async def list_loras(request: Request):  # noqa: ARG001
    """Return discovered LoRA weights as a sorted list.

    round7:必须声明 `request` 参数 —— @cached 靠 kwargs.get("request") 取 Request 建缓存
    key,否则直接 fall-through 不缓存(ETag/304/30s body 缓存全失效,每次全量重下)。
    对齐 services.py/workflows.py 的写法。

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
