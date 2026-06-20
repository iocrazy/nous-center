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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_serializer, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from src.api.deps_admin import require_admin
import logging

from src.api.response_cache import cached, invalidate
from src.models.database import get_async_session
from src.models.schemas import ExposedParam
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow
from src.services.service_models import extract_service_models

router = APIRouter(prefix="/api/v1", tags=["services"])

logger = logging.getLogger(__name__)

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


class ServiceModelRef(BaseModel):
    """Static ref to a model/component a service depends on. Live load-state
    is overlaid client-side (see src/services/service_models.py)."""
    kind: str  # 'component' | 'engine'
    role: str | None = None  # diffusion_models|clip|vae|checkpoint|llm|tts
    label: str
    file: str | None = None  # component abs path (matched by file)
    engine_key: str | None = None  # registry engine key (matched by name)


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
    workflow_name: str | None = None  # join from workflows.name；UI 用来显示"来自 Workflow · {name}"
    snapshot_hash: str | None = None
    snapshot_schema_version: int = 1
    version: int = 1
    # 该服务工作流依赖的模型/组件(静态枚举;加载状态前端实时叠加)。
    models: list[ServiceModelRef] = []
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
    # 改名(= 改 model 路由键)。⚠️ 用旧 model 名调用的客户端会失效(404),UI 给提示。
    # 格式同发布(^[a-z][a-z0-9-]{1,62}$),唯一;grant/用量靠 service_id 不受影响。
    name: str | None = None
    # 服务页「应用编辑」tab 改暴露字段(逐 widget 表单配置)就地落库。映射改了
    # 不动 snapshot 本体,故不 bump snapshot_hash/version(见 spec 2026-06-09 R3:
    # 改 active 服务的对外 schema 有契约风险,UI 给提示,单管理员 infra 允许)。
    exposed_inputs: list[ExposedParam] | None = None
    exposed_outputs: list[ExposedParam] | None = None


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


_METER_DIM_BY_CATEGORY = {"llm": "tokens", "tts": "chars", "vl": "calls", "image": "images"}


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
    dependencies=[Depends(require_admin)],
)
@cached("services", ttl=30)
async def list_services(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    category: str | None = None,
    status: str | None = None,
):
    # LEFT JOIN workflows 把 name 一起带回来 — UI 服务卡片显示
    # "来自 Workflow · {name} #{id}"。trivial workflow 名是 "trivial:{svc}"，
    # 用户能立刻看出是快速开通生成的 vs 真实 workflow。
    stmt = (
        select(ServiceInstance, Workflow.name.label("workflow_name"))
        .outerjoin(Workflow, Workflow.id == ServiceInstance.workflow_id)
        # un-defer snapshot so we can enumerate each service's model deps for
        # the list-card「已加载 X/Y」badge. Single-admin infra has a handful of
        # services + the response is @cached(ttl=30), so the cost is absorbed.
        .options(undefer(ServiceInstance.workflow_snapshot))
        .order_by(ServiceInstance.created_at.desc())
    )
    if category:
        stmt = stmt.where(ServiceInstance.category == category)
    if status:
        stmt = stmt.where(ServiceInstance.status == status)
    rows = (await session.execute(stmt)).all()
    out = []
    for svc, wf_name in rows:
        item = ServiceOut.model_validate(svc)
        item.workflow_name = wf_name
        item.models = [ServiceModelRef(**m) for m in extract_service_models(svc.workflow_snapshot)]
        out.append(item)
    return out


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
        select(ServiceInstance, Workflow.name.label("workflow_name"))
        .outerjoin(Workflow, Workflow.id == ServiceInstance.workflow_id)
        .options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
        .where(ServiceInstance.id == service_id)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        raise HTTPException(404, detail="service not found")
    svc, wf_name = row
    out = ServiceDetailOut.model_validate(svc)
    out.workflow_name = wf_name
    out.models = [ServiceModelRef(**m) for m in extract_service_models(svc.workflow_snapshot)]
    return out


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
    # round4 #6:名字唯一性预检(上面 line 224)是 TOCTOU —— 并发同名两个请求都过预检,
    # 第二个 commit 抛 IntegrityError。早先无 try/except → 500。捕获 → 409,与预检口径一致。
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, detail=f"service name '{body.name}' already exists")
    # Force-load deferred columns before serializing — pydantic from_attributes
    # would otherwise trigger sync lazy load inside an async context.
    await session.refresh(
        svc,
        attribute_names=["workflow_snapshot", "exposed_inputs", "exposed_outputs"],
    )
    # Cross-resource: quick-provision creates BOTH a workflow row and a service row.
    invalidate("services", "workflows")
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
    edits_exposed = body.exposed_inputs is not None or body.exposed_outputs is not None
    # workflow_snapshot / exposed_* are deferred() — touching them in the async
    # path without undefer raises MissingGreenlet (lazy load = sync IO). Only
    # pull them when this PATCH actually edits the exposed schema.
    stmt = select(ServiceInstance).where(ServiceInstance.id == service_id)
    if edits_exposed:
        stmt = stmt.options(
            undefer(ServiceInstance.workflow_snapshot),
            undefer(ServiceInstance.exposed_inputs),
            undefer(ServiceInstance.exposed_outputs),
        )
    svc = await session.scalar(stmt)
    if svc is None:
        raise HTTPException(404, detail="service not found")
    if body.status is not None:
        svc.status = body.status
    if body.name is not None and body.name != svc.name:
        # 校验格式 + 唯一(预检;commit 处再兜 IntegrityError 防并发 TOCTOU)。
        try:
            _validate_name(body.name)
        except ValueError as e:
            raise HTTPException(422, detail=str(e))
        taken = await session.scalar(
            select(ServiceInstance.id).where(
                ServiceInstance.name == body.name, ServiceInstance.id != service_id),
        )
        if taken is not None:
            raise HTTPException(409, detail=f"service name '{body.name}' already exists")
        svc.name = body.name
    if edits_exposed:
        _validate_exposed_against_snapshot(
            dict(svc.workflow_snapshot or {}),
            body.exposed_inputs,
            body.exposed_outputs,
        )
        if body.exposed_inputs is not None:
            svc.exposed_inputs = [p.model_dump(exclude_none=True) for p in body.exposed_inputs]
        if body.exposed_outputs is not None:
            svc.exposed_outputs = [p.model_dump(exclude_none=True) for p in body.exposed_outputs]
    try:
        await session.commit()
    except IntegrityError:  # 改名并发撞唯一约束(TOCTOU 兜底)
        await session.rollback()
        raise HTTPException(409, detail=f"service name '{body.name}' already exists")
    await session.refresh(svc)
    invalidate("services")
    return svc


def _validate_exposed_against_snapshot(
    snapshot: dict,
    inputs: list[ExposedParam] | None,
    outputs: list[ExposedParam] | None,
) -> None:
    """Same hard contract as publish: every exposed.node_id must resolve in
    the frozen snapshot, and image-node outputs may only reference fields the
    node actually emits. Imported lazily — workflow_publish imports from this
    module, so a top-level import would be circular."""
    from src.api.routes.workflow_publish import (
        _IMAGE_NODE_TYPES,
        _IMAGE_OUTPUT_FIELDS,
        _node_ids,
        _node_types_by_id,
    )

    valid_ids = _node_ids(snapshot)
    types_by_id = _node_types_by_id(snapshot)
    for kind, params in (("input", inputs), ("output", outputs)):
        for p in params or []:
            if str(p.node_id) not in valid_ids:
                raise HTTPException(
                    422,
                    detail=(
                        f"exposed {kind} references node_id {p.node_id!r} "
                        f"that does not exist in the workflow snapshot"
                    ),
                )
    for p in outputs or []:
        node_type = types_by_id.get(str(p.node_id))
        if node_type not in _IMAGE_NODE_TYPES:
            continue
        field = p.input_name
        if field and field not in _IMAGE_OUTPUT_FIELDS:
            raise HTTPException(
                422,
                detail=(
                    f"exposed output input_name={field!r} is not emitted by "
                    f"{node_type}; allowed: {sorted(_IMAGE_OUTPUT_FIELDS)}"
                ),
            )


@router.delete(
    "/services/{service_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_service(
    service_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    svc = await session.get(ServiceInstance, service_id)
    if svc is None:
        raise HTTPException(404, detail="service not found")
    wf_id = svc.workflow_id
    await session.delete(svc)
    await session.flush()

    # 删服务后:若源工作流已无任何关联服务,把 status 退回 "draft"(镜像 /unpublish)。
    # 之前漏了这步 → 工作流卡在 status="published" 但无关联服务,卡片头部"已发布"绿徽章
    # 与底部"未关联服务"+发布按钮语义冲突(用户反馈)。顺带卸掉该工作流的模型引用。
    if wf_id is not None:
        still = await session.execute(
            select(ServiceInstance.id).where(ServiceInstance.workflow_id == wf_id).limit(1)
        )
        if still.first() is None:
            wf = await session.get(Workflow, wf_id)
            if wf is not None and wf.status == "published":
                wf.status = "draft"
                model_mgr = getattr(request.app.state, "model_manager", None)
                if model_mgr is not None:
                    try:
                        for dep in model_mgr.get_model_dependencies(
                            {"nodes": wf.nodes, "edges": wf.edges}
                        ):
                            model_mgr.remove_reference(dep["key"], str(wf.id))
                            await model_mgr.unload_model(dep["key"])
                    except Exception as e:  # 模型清理失败不应阻断删除
                        logger.warning(
                            "delete_service: model deref failed for workflow %s: %s", wf_id, e
                        )

    await session.commit()
    # 工作流 status 可能翻回 draft → workflows 列表缓存也要失效。
    invalidate("services", "workflows")
