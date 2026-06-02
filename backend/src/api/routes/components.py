"""GET /api/v1/components + POST /scan — file index for PR-4 loader nodes.

GET ?role=diffusion_models|clip|vae|loras → that role's files.
GET (no role)                 → full index across all roles.
POST /scan                    → admin-only re-glob + cache invalidate + WS broadcast.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps_admin import require_admin
from src.api.websocket import ws_manager
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


@router.get("/seedvr2-dit")
async def list_seedvr2_dit_models():
    """SeedVR2 DiT 白名单 + 磁盘状态(present/size)。给节点 seedvr2_model_select widget「混合」
    展示:盘上有的标已就绪、白名单其余标可下载(NumZ 选了从 HF 自动下)。

    跟 component_scanner 的 role 不同:SeedVR2 不能「盘上啥 safetensors 都列」(NumZ 按白名单+
    sha256 加载,盘上的非白名单旧变体引擎不认),所以这里走白名单交叉磁盘,不复用 scan_components。
    """
    from src.services.inference.image_seedvr2 import seedvr2_dit_models_with_disk_status
    return {"models": seedvr2_dit_models_with_disk_status()}


@router.post("/scan", dependencies=[Depends(require_admin)])
async def rescan_components():
    """Re-glob the model dirs + invalidate cache + broadcast WS event."""
    invalidate_component_cache()
    index = get_component_index()
    total = sum(len(v) for v in index.values())
    # Piggyback the existing /ws/models channel with a sentinel model_id rather
    # than open a dedicated WS channel. Frontend (PR-5) treats model_id
    # "__components__" as the "rescan finished, refresh dropdowns" signal.
    await ws_manager.broadcast_model_status("__components__", "index_changed", f"{total} files")
    return {"status": "rescanned", "total": total}
