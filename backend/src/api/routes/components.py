"""GET /api/v1/components + POST /scan — file index for PR-4 loader nodes.

GET ?role=unet|clip|vae|loras → that role's files.
GET (no role)                 → full index across all roles.
POST /scan                    → admin-only re-glob + cache invalidate + WS broadcast.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps_admin import require_admin
from src.services.component_scanner import (
    ROLE_DIRS,
    get_component_index,
    invalidate_component_cache,
    scan_components,
)

router = APIRouter(prefix="/api/v1/components", tags=["components"])


@router.get("")
async def list_components(role: str | None = Query(default=None)):
    """List component files. With ?role= → that role; without → full index."""
    if role is None:
        return {"index": get_component_index()}
    if role not in ROLE_DIRS:
        raise HTTPException(400, detail=f"unknown role {role!r}; expected one of {list(ROLE_DIRS)}")
    return {"components": scan_components(role)}


@router.post("/scan", dependencies=[Depends(require_admin)])
async def rescan_components():
    """Re-glob the model dirs + invalidate cache + broadcast WS event."""
    invalidate_component_cache()
    index = get_component_index()
    total = sum(len(v) for v in index.values())
    from src.api.websocket import ws_manager
    await ws_manager.broadcast_model_status("__components__", "index_changed", f"{total} files")
    return {"status": "rescanned", "total": total}
