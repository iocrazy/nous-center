"""v3 services CRUD + quick-provision path.

The two creation paths in v3 produce the same row in `service_instances`:
this module owns "quick-provision" (auto-generated trivial workflow);
`routes/workflow_publish.py` owns "publish from workflow".

Endpoints:
  GET    /api/v1/services                 — list (admin)
  GET    /api/v1/services/{id}            — detail with snapshot (admin)
  POST   /api/v1/services/quick-provision — quick-provision (admin)
  PATCH  /api/v1/services/{id}            — status lifecycle (admin)
  DELETE /api/v1/services/{id}            — delete (admin)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_serializer, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.deps_admin import require_admin
from src.models.database import get_async_session
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1", tags=["services"])

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")


def _validate_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError(
            "service name must match ^[a-z][a-z0-9-]{1,62}$ "
            "(start with a-z, then a-z/0-9/-, total 2-63 chars)",
        )
    return name


def _snapshot_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------- Pydantic shapes ----------


class ServiceOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    type: str
    status: str
    source_type: str
    source_id: int | None = None
    source_name: str | None = None
    category: str | None = None
    meter_dim: str | None = None
    workflow_id: int | None = None
    snapshot_hash: str | None = None
    snapshot_schema_version: int = 1
    version: int = 1
    created_at: datetime
    updated_at: datetime

    @field_serializer("id", "source_id", "workflow_id", when_used="json")
    def _to_str(self, v: int | None) -> str | None:
        return str(v) if v is not None else None


class ServiceDetailOut(ServiceOut):
    workflow_snapshot: dict
    exposed_inputs: list
    exposed_outputs: list


class ServicePatch(BaseModel):
    status: Literal["active", "paused", "deprecated", "retired"] | None = None


class QuickProvisionBody(BaseModel):
    name: str
    category: Literal["llm", "tts", "vl"]
    engine: str = Field(..., description="Engine key, e.g. 'qwen3-8b' / 'cosyvoice2'")
    label: str = ""
    params: dict[str, Any] = {}
    """Free-form engine knobs (system_prompt, temperature, voice, …)."""

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_name(v)


# ---------- Helpers ----------


_METER_DIM_BY_CATEGORY = {"llm": "tokens", "tts": "chars", "vl": "calls"}


def _trivial_workflow_for(category: str, engine: str, params: dict[str, Any]) -> dict:
    """Build the minimum DAG that backs a quick-provisioned service.

    The shape is intentionally tiny — three nodes (input, engine call,
    output) with a fixed wiring. The executor doesn't need to understand
    it during PR-A; the workflow exists so PR-B's "源 Workflow" link in
    the service detail page has something to point at.
    """
    return {
        "schema": "comfy/api-1",
        "nodes": {
            "in_1": {
                "class_type": "PrimitiveInput",
                "inputs": {"value": params.get("default_input", "")},
                "_meta": {"role": "exposed_input"},
            },
            "engine_1": {
                "class_type": f"{category.upper()}Engine",
                "inputs": {
                    "engine": engine,
                    "prompt": ["in_1", 0],
                    **{k: v for k, v in params.items() if k != "default_input"},
                },
            },
            "out_1": {
                "class_type": "PrimitiveOutput",
                "inputs": {"value": ["engine_1", 0]},
                "_meta": {"role": "exposed_output"},
            },
        },
    }


# ---------- Routes ----------


@router.get(
    "/services",
    response_model=list[ServiceOut],
    dependencies=[Depends(require_admin)],
)
async def list_services(
    session: AsyncSession = Depends(get_async_session),
    category: str | None = None,
    status: str | None = None,
):
    stmt = select(ServiceInstance).order_by(ServiceInstance.created_at.desc())
    if category:
        stmt = stmt.where(ServiceInstance.category == category)
    if status:
        stmt = stmt.where(ServiceInstance.status == status)
    rows = (await session.execute(stmt)).scalars().all()
    return rows


@router.get(
    "/services/{service_id}",
    response_model=ServiceDetailOut,
    dependencies=[Depends(require_admin)],
)
async def get_service(
    service_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        select(ServiceInstance)
        .options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
        .where(ServiceInstance.id == service_id)
    )
    svc = (await session.execute(stmt)).scalar_one_or_none()
    if svc is None:
        raise HTTPException(404, detail="service not found")
    return svc


@router.post(
    "/services/quick-provision",
    response_model=ServiceDetailOut,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
async def quick_provision(
    body: QuickProvisionBody,
    session: AsyncSession = Depends(get_async_session),
):
    # Name uniqueness is also enforced by a UNIQUE constraint; this gives a
    # 409 instead of a 500 on collision.
    existing = await session.scalar(
        select(ServiceInstance).where(ServiceInstance.name == body.name)
    )
    if existing is not None:
        raise HTTPException(409, detail=f"service name '{body.name}' already exists")

    snapshot = _trivial_workflow_for(body.category, body.engine, body.params)

    workflow = Workflow(
        name=f"trivial:{body.name}",
        description=f"auto-generated for service {body.name}",
        nodes=[],
        edges=[],
        is_template=False,
        status="active",
        auto_generated=True,
    )
    session.add(workflow)
    await session.flush()

    svc = ServiceInstance(
        name=body.name,
        type="inference",
        status="active",
        source_type="workflow",
        source_id=workflow.id,
        source_name=body.engine,
        category=body.category,
        meter_dim=_METER_DIM_BY_CATEGORY.get(body.category, "calls"),
        workflow_id=workflow.id,
        workflow_snapshot=snapshot,
        snapshot_hash=_snapshot_hash(snapshot),
        snapshot_schema_version=1,
        version=1,
        exposed_inputs=[
            {
                "key": "input",
                "label": body.label or "Input",
                "node_id": "in_1",
                "input_name": "value",
                "type": "string",
                "required": True,
            }
        ],
        exposed_outputs=[
            {
                "key": "output",
                "label": "Output",
                "node_id": "out_1",
                "input_name": "value",
                "type": "string",
            }
        ],
    )
    session.add(svc)
    await session.flush()

    workflow.generated_for_service_id = svc.id
    await session.commit()
    # Force-load deferred columns before serializing — pydantic from_attributes
    # would otherwise trigger sync lazy load inside an async context.
    await session.refresh(
        svc,
        attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
    )
    return svc


@router.patch(
    "/services/{service_id}",
    response_model=ServiceOut,
    dependencies=[Depends(require_admin)],
)
async def patch_service(
    service_id: int,
    body: ServicePatch,
    session: AsyncSession = Depends(get_async_session),
):
    svc = await session.get(ServiceInstance, service_id)
    if svc is None:
        raise HTTPException(404, detail="service not found")
    if body.status is not None:
        svc.status = body.status
    await session.commit()
    await session.refresh(svc)
    return svc


@router.delete(
    "/services/{service_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_service(
    service_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    svc = await session.get(ServiceInstance, service_id)
    if svc is None:
        raise HTTPException(404, detail="service not found")
    await session.delete(svc)
    await session.commit()
