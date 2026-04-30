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
from src.api.routes.services import NAME_RE, ServiceDetailOut
from src.models.database import get_async_session
from src.models.schemas import ExposedParam
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1", tags=["workflow-publish"])


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

    svc = ServiceInstance(
        name=body.name,
        type="inference",
        status="active",
        source_type="workflow",
        source_id=workflow_id,
        category=body.category or "app",
        meter_dim=body.meter_dim or "calls",
        workflow_id=workflow_id,
        workflow_snapshot=snapshot,
        exposed_inputs=[p.model_dump(exclude_none=True) for p in body.exposed_inputs],
        exposed_outputs=[p.model_dump(exclude_none=True) for p in body.exposed_outputs],
        snapshot_hash=snapshot_hash,
        snapshot_schema_version=1,
        version=1,
    )
    session.add(svc)
    # Flip the source workflow to "published" so the list view's badge + tab
    # filter reflect reality. Without this the only signal that a workflow
    # has been published was the existence of a ServiceInstance row, and the
    # two could (and did) drift — see the orphan rows cleared in the
    # accompanying data fix.
    wf.status = "published"
    await session.commit()
    await session.refresh(
        svc,
        attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
    )
    invalidate("services", "workflows")
    return svc
