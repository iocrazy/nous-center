"""Workflow CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import WorkflowCreate, WorkflowUpdate, WorkflowOut
from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow
from src.services import model_scheduler

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowOut, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = Workflow(**body.model_dump())
    session.add(wf)
    await session.commit()
    await session.refresh(wf)
    return wf


@router.get("", response_model=list[WorkflowOut])
async def list_workflows(
    is_template: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Workflow).order_by(Workflow.updated_at.desc())
    if is_template is not None:
        stmt = stmt.where(Workflow.is_template == is_template)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@router.patch("/{workflow_id}", response_model=WorkflowOut)
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
    return wf


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    await session.delete(wf)
    await session.commit()


@router.post("/{workflow_id}/publish")
async def publish_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
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
        instance.params_override = {"nodes": wf.nodes, "edges": wf.edges}
        instance.status = "active"
    else:
        instance = ServiceInstance(
            source_type="workflow",
            source_id=wf.id,
            name=wf.name,
            type="workflow",
            params_override={"nodes": wf.nodes, "edges": wf.edges},
        )
        session.add(instance)
        await session.flush()
        instance.endpoint_path = f"/v1/instances/{instance.id}/run"

    # Auto-load model dependencies
    deps = model_scheduler.get_model_dependencies(
        {"nodes": wf.nodes, "edges": wf.edges}
    )
    for dep in deps:
        try:
            await model_scheduler.load_model(dep["key"])
            await model_scheduler.add_reference(dep["key"], str(wf.id))
        except Exception as e:
            raise HTTPException(503, f"无法加载模型 {dep['key']}: {e}")

    wf.status = "published"
    await session.commit()
    await session.refresh(instance)

    return {
        "instance_id": str(instance.id),
        "endpoint": instance.endpoint_path,
    }


@router.post("/{workflow_id}/unpublish")
async def unpublish_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
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
    deps = model_scheduler.get_model_dependencies(
        {"nodes": wf.nodes, "edges": wf.edges}
    )
    for dep in deps:
        await model_scheduler.remove_reference(dep["key"], str(wf.id))
        await model_scheduler.unload_model(dep["key"])

    wf.status = "draft"
    await session.commit()
    return {"status": "unpublished"}


@router.post("/execute")
async def execute_workflow_direct(
    body: dict,
    session: AsyncSession = Depends(get_async_session),
):
    """Execute a workflow directly without publishing. Used by frontend Run for plugin nodes."""
    import time
    from src.services.workflow_executor import WorkflowExecutor, ExecutionError
    from src.models.execution_task import ExecutionTask

    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    if not nodes:
        raise HTTPException(400, "Workflow is empty")

    # Create task record
    task = ExecutionTask(
        workflow_name=body.get("name", "直接执行"),
        status="running",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    start = time.monotonic()
    executor = WorkflowExecutor({"nodes": nodes, "edges": edges})

    try:
        result = await executor.execute()
        elapsed = int((time.monotonic() - start) * 1000)
        task.status = "completed"
        task.result = result
        task.duration_ms = elapsed
        task.nodes_done = len(nodes)
        task.current_node = None
    except ExecutionError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        task.status = "failed"
        task.error = str(e)
        task.duration_ms = elapsed
        await session.commit()
        raise HTTPException(500, str(e))

    await session.commit()
    return result
