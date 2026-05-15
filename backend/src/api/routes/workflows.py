"""Workflow CRUD routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.api.response_cache import cached, invalidate
from src.models.database import get_async_session
from src.models.schemas import WorkflowCreate, WorkflowUpdate, WorkflowOut
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowOut, status_code=201, dependencies=[Depends(require_admin)])
async def create_workflow(
    body: WorkflowCreate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = Workflow(**body.model_dump())
    session.add(wf)
    await session.commit()
    await session.refresh(wf)
    invalidate("workflows")
    return wf


@router.get("", response_model=list[WorkflowOut])
@cached("workflows", ttl=30)
async def list_workflows(
    request: Request,
    is_template: bool | None = None,
    auto_generated: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Workflow).order_by(Workflow.updated_at.desc())
    if is_template is not None:
        stmt = stmt.where(Workflow.is_template == is_template)
    if auto_generated is not None:
        stmt = stmt.where(Workflow.auto_generated == auto_generated)
    result = await session.execute(stmt)
    # Force Pydantic serialization NOW (not via response_model alone) so the
    # @cached layer stores the JSON-shaped dicts with id-as-string. Without
    # this, ORM Workflow objects flow through cached() and then FastAPI
    # serializes them after — losing field_serializer("id") and shipping
    # snowflake IDs as raw numbers (which JS truncates to 17 digits, making
    # every workflow look like the same one to the router).
    rows = result.scalars().all()
    return [WorkflowOut.model_validate(r).model_dump(mode="json") for r in rows]


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@router.patch("/{workflow_id}", response_model=WorkflowOut, dependencies=[Depends(require_admin)])
async def update_workflow(
    workflow_id: int,
    body: WorkflowUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(wf, key, value)
    await session.commit()
    await session.refresh(wf)
    invalidate("workflows")
    return wf


@router.delete("/{workflow_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    await session.delete(wf)
    await session.commit()
    invalidate("workflows")


# NOTE: the legacy POST /api/v1/workflows/{id}/publish handler was removed
# in PR-A (v3 IA rebuild). The v3 publish path lives in
# src/api/routes/workflow_publish.py and produces a service with a frozen
# snapshot + exposed schema instead of just storing nodes in
# params_override. PR-B will re-wire the frontend "publish" button.


@router.post("/{workflow_id}/unpublish", dependencies=[Depends(require_admin)])
async def unpublish_workflow(
    workflow_id: int,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    model_mgr = request.app.state.model_manager
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")

    stmt = select(ServiceInstance).where(
        ServiceInstance.source_type == "workflow",
        ServiceInstance.source_id == workflow_id,
    )
    result = await session.execute(stmt)
    instance = result.scalar_one_or_none()
    if instance:
        instance.status = "inactive"

    # Remove model references and attempt unload
    deps = model_mgr.get_model_dependencies(
        {"nodes": wf.nodes, "edges": wf.edges}
    )
    for dep in deps:
        model_mgr.remove_reference(dep["key"], str(wf.id))
        await model_mgr.unload_model(dep["key"])

    wf.status = "draft"
    await session.commit()
    # Cross-resource invalidation: unpublish flips workflow.status AND
    # service.status, so the services list cache must drop too.
    invalidate("workflows", "services")
    return {"status": "unpublished"}


@router.post("/execute", status_code=202)
async def execute_workflow_direct(
    body: dict,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """入队一个 workflow 直接执行（D17 纯异步）：建 task → 后台跑 → 立即返回 202 + task_id。

    客户端拿结果：轮询 GET /api/v1/tasks/{task_id} 或订阅 WS /ws/workflow/{task_id}。
    """
    import asyncio

    from src.models.execution_task import ExecutionTask
    from src.services.workflow_runner import run_workflow_task

    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    if not nodes:
        raise HTTPException(400, "Workflow is empty")

    task = ExecutionTask(
        workflow_name=body.get("name", "直接执行"),
        status="queued",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # channel_id：D17 之后客户端订阅 WS 用 task_id 作为 channel；兼容旧的
    # 显式 channel_id（前端在 POST 前先开 WS 的老流程）。
    channel_id = body.get("channel_id") or str(task.id)
    runner_client = getattr(request.app.state, "runner_client", None)

    asyncio.create_task(run_workflow_task(
        task.id, {"nodes": nodes, "edges": edges},
        runner_client=runner_client, channel_id=channel_id,
    ))
    return {"task_id": str(task.id), "status": "queued", "channel_id": channel_id}
