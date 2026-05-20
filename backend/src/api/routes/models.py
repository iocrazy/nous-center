"""GET /api/v1/models — raw scanner output.

Sibling to /api/v1/engines. Where /engines returns *loadable* yaml-curated
entries enriched with load_status + metadata, /models returns the unfiltered
scanner walk: yaml entries + any auto-detected dirs on disk, with their
preset `files{}` block when declared.

V1' Lane C component-node executors will read this to populate dropdowns
(LoadCheckpoint picks from `type == 'image'`; the component-file dropdowns
go through a separate scanner endpoint added in a later PR).
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from src.api.deps_admin import require_admin
from src.api.response_cache import cached
from src.services.inference.component_spec import ComponentSpec
from src.services.model_scanner import scan_models

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
@cached("models", ttl=30)
async def list_models():
    """Return every model the scanner sees.

    Response shape: ``{"models": [{...}, ...]}`` keyed list (NOT a dict),
    sorted by id for stable ETags. The cache is invalidated alongside
    /engines on /engines/scan and similar mutating endpoints — see
    `main.py:_invalidate("models", "engines")`.
    """
    configs = scan_models()
    items = [{"id": k, **v} for k, v in sorted(configs.items())]
    return {"models": items}


@router.get("/components/state")
async def get_components_state(request: Request, keys: str | None = Query(default=None)):
    """Batch component load-state (spec §6.3). `keys` = comma-separated
    component_state_key list; omitted → all known. Unknown keys → 'cold'."""
    reg = getattr(request.app.state, "component_state_registry", None)
    if reg is None:
        return {"components": []}
    if keys:
        wanted = [k for k in keys.split(",") if k]
        return {"components": reg.query(wanted)}
    return {"components": reg.all()}


@router.post("/components/preload", status_code=202, dependencies=[Depends(require_admin)])
async def preload_components(request: Request, body: dict = Body(...)):
    """Batch-warm a unet+clip+vae combo on the image runner (spec §6.2). Returns
    202 + task_id immediately; state arrives via /ws/models component_state_changed."""
    raw = body.get("components") or {}
    missing = {"unet", "clip", "vae"} - set(raw)
    if missing:
        raise HTTPException(422, f"missing component kinds: {sorted(missing)}")
    try:
        components = {k: ComponentSpec(**raw[k]).model_dump() for k in ("unet", "clip", "vae")}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"invalid component spec: {e}") from e

    client = (getattr(request.app.state, "runner_clients", {}) or {}).get("image")
    if client is None or not getattr(client, "_connected", True):
        raise HTTPException(503, "image runner not available")

    task_id = int(time.time() * 1000) % (2**31)
    pipeline_class = str(body.get("pipeline_class") or "Flux2KleinPipeline")
    await client.preload_components(task_id=task_id, components=components, pipeline_class=pipeline_class)
    return {"task_id": task_id, "status": "accepted"}
