"""POST /api/v1/workflows/{id}/publish — freeze a workflow into a service.

The publish flow is the second creation path for a service (the first
being quick-provision in routes/services.py). It snapshots the source
workflow and stores it on a fresh `service_instances` row, with the
caller-supplied `exposed_inputs` / `exposed_outputs` defining the
external schema.

Per the v3 plan we MUST validate that every `exposed.node_id` referenced
by the schema actually exists in the snapshot — silently shipping a
service whose schema points at deleted nodes would route caller payloads
into a void.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.api.response_cache import invalidate
from src.api.routes.services import NAME_RE, ServiceDetailOut, _METER_DIM_BY_CATEGORY
from src.models.database import get_async_session
from src.models.schemas import ExposedParam
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1", tags=["workflow-publish"])

# Output fields image-producing nodes emit — exposed_outputs pointing at
# any node in `_IMAGE_NODE_TYPES` may only reference one of these.
# Anything else is either a typo (image_url vs image_uri) or a mistake
# about the envelope shape; either way, surfacing 422 at publish time
# beats a 500 at runtime when the caller hits the service.
#
# Both nodes (the V0 integrated image_generate and the V1' Lane C
# component-path terminus flux2_vae_decode) emit the same {image_url,
# media_type, width, height, image_uuid, image_expires} bundle —
# image_generate additionally carries steps/seed/cfg/loras/duration_ms
# metadata. We use the union as the allowlist; the field check is
# inclusive across both terminus types.
_IMAGE_OUTPUT_FIELDS = {
    "image_url",
    "image",          # base64 fallback for dev mode
    "image_uuid",
    "image_expires",
    "media_type",
    "width",
    "height",
    "steps",
    "seed",
    "loras",
    "duration_ms",
}

_IMAGE_NODE_TYPES = {"image_generate", "flux2_vae_decode"}


def _snapshot_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _node_ids(snapshot: dict) -> set[str]:
    """Pull node ids out of either the api-style dict or the editor-style list."""
    nodes = snapshot.get("nodes")
    if isinstance(nodes, dict):
        return {str(k) for k in nodes.keys()}
    if isinstance(nodes, list):
        return {str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("id") is not None}
    return set()


def _node_types_by_id(snapshot: dict) -> dict[str, str]:
    """Map node_id → class_type for either snapshot shape."""
    out: dict[str, str] = {}
    nodes = snapshot.get("nodes")
    if isinstance(nodes, dict):
        for nid, node in nodes.items():
            ct = node.get("class_type") if isinstance(node, dict) else None
            if ct:
                out[str(nid)] = str(ct)
    elif isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            ct = n.get("class_type") or n.get("type")
            if nid is not None and ct:
                out[str(nid)] = str(ct)
    return out


def _detect_category(snapshot: dict) -> str | None:
    """Heuristic per-modality detection from the snapshot's node types.

    Returned value drops into ServiceInstance.category + meter_dim. We
    only cover image here in PR-7 — LLM/TTS/VL flow through the explicit
    body.category path that quick-provision already controls.
    """
    types = set(_node_types_by_id(snapshot).values())
    if types & _IMAGE_NODE_TYPES:
        return "image"
    return None


class PublishBody(BaseModel):
    name: str
    label: str = ""
    category: str | None = None  # llm|tts|vl|app
    meter_dim: str | None = None
    exposed_inputs: list[ExposedParam] = []
    exposed_outputs: list[ExposedParam] = []

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(
                "service name must match ^[a-z][a-z0-9-]{1,62}$",
            )
        return v


def _build_snapshot(wf: Workflow) -> dict[str, Any]:
    """Render the workflow's working state into the api-shape we freeze.

    Editor stores nodes as a list with explicit ids; api-shape is a dict
    keyed by node id. We always emit api-shape so consumers (executor,
    schema validator) only have to handle one form.
    """
    nodes_dict: dict[str, Any] = {}
    for node in (wf.nodes or []):
        nid = node.get("id")
        if nid is None:
            continue
        nodes_dict[str(nid)] = {
            "class_type": node.get("type") or node.get("class_type"),
            "inputs": node.get("data", node.get("inputs", {})),
            "_meta": node.get("meta", {}),
        }
    return {
        "schema": "comfy/api-1",
        "nodes": nodes_dict,
        "edges": wf.edges or [],
    }


@router.post(
    "/workflows/{workflow_id}/publish",
    response_model=ServiceDetailOut,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
async def publish_workflow(
    workflow_id: int,
    body: PublishBody,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if wf is None:
        raise HTTPException(404, detail="workflow not found")
    if wf.auto_generated:
        # Trivial workflows are owned by their service. Re-publishing one
        # would create a fork that contradicts "service = published wf".
        raise HTTPException(
            409,
            detail="cannot publish an auto-generated workflow; edit the source instead",
        )

    snapshot = _build_snapshot(wf)
    valid_ids = _node_ids(snapshot)
    types_by_id = _node_types_by_id(snapshot)

    # Hard contract: every exposed.node_id MUST resolve.
    for kind, params in (("input", body.exposed_inputs), ("output", body.exposed_outputs)):
        for p in params:
            if str(p.node_id) not in valid_ids:
                raise HTTPException(
                    422,
                    detail=(
                        f"exposed {kind} references node_id {p.node_id!r} "
                        f"that does not exist in the workflow snapshot"
                    ),
                )

    # Image envelope guard: a typo in input_name ("image_uri" vs
    # "image_url") would silently publish a service whose response payload
    # is `null`. Reject at publish time so the caller sees the mistake
    # before any downstream consumer integrates against it.
    for p in body.exposed_outputs:
        node_type = types_by_id.get(str(p.node_id))
        if node_type not in _IMAGE_NODE_TYPES:
            continue
        field = p.input_name
        if field and field not in _IMAGE_OUTPUT_FIELDS:
            raise HTTPException(
                422,
                detail=(
                    f"exposed output input_name={field!r} is not emitted by "
                    f"{node_type}; allowed: "
                    f"{sorted(_IMAGE_OUTPUT_FIELDS)}"
                ),
            )

    # Name collision: bump version on existing-name re-publish? v3 says
    # "禁止重命名，只允许 deprecate + new name". So we 409 on name collision
    # — callers are expected to either pick a new name or PATCH the old
    # service to `deprecated` first.
    existing = await session.scalar(
        select(ServiceInstance).where(ServiceInstance.name == body.name)
    )
    if existing is not None:
        raise HTTPException(409, detail=f"service name '{body.name}' already exists")

    snapshot_hash = _snapshot_hash(snapshot)

    # When the caller doesn't lock the category, infer from snapshot
    # contents so meter_dim picks up the right unit (image → "images").
    detected_category = _detect_category(snapshot)
    resolved_category = body.category or detected_category or "app"
    resolved_meter = (
        body.meter_dim
        or _METER_DIM_BY_CATEGORY.get(resolved_category)
        or "calls"
    )

    svc = ServiceInstance(
        name=body.name,
        type="inference",
        status="active",
        source_type="workflow",
        source_id=workflow_id,
        category=resolved_category,
        meter_dim=resolved_meter,
        workflow_id=workflow_id,
        workflow_snapshot=snapshot,
        exposed_inputs=[p.model_dump(exclude_none=True) for p in body.exposed_inputs],
        exposed_outputs=[p.model_dump(exclude_none=True) for p in body.exposed_outputs],
        snapshot_hash=snapshot_hash,
        snapshot_schema_version=1,
        version=1,
    )
    session.add(svc)
    await session.commit()
    await session.refresh(
        svc,
        attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
    )
    # Publish creates a service AND flips workflow.status="published" downstream;
    # both list caches must drop.
    invalidate("services", "workflows")
    return svc
