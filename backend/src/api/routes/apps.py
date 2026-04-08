"""WorkflowApp publish and execute routes."""

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.models.database import get_async_session
from src.models.schemas import WorkflowAppPublish, WorkflowAppOut
from src.models.workflow import Workflow
from src.models.workflow_app import WorkflowApp

router = APIRouter(tags=["apps"])


@router.post(
    "/api/v1/workflows/{workflow_id}/publish-app",
    response_model=WorkflowAppOut,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
async def publish_app(
    workflow_id: int,
    body: WorkflowAppPublish,
    session: AsyncSession = Depends(get_async_session),
):
    # Fetch the workflow
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")

    # Check name uniqueness
    existing = await session.scalar(
        select(WorkflowApp).where(WorkflowApp.name == body.name)
    )
    if existing:
        raise HTTPException(409, f"App name '{body.name}' already exists")

    app_obj = WorkflowApp(
        name=body.name,
        display_name=body.display_name,
        description=body.description,
        workflow_id=workflow_id,
        workflow_snapshot={"nodes": wf.nodes, "edges": wf.edges},
        exposed_inputs=[p.model_dump() for p in body.exposed_inputs],
        exposed_outputs=[p.model_dump() for p in body.exposed_outputs],
    )
    session.add(app_obj)
    await session.commit()
    await session.refresh(app_obj)
    return app_obj


@router.get("/api/v1/apps", response_model=list[WorkflowAppOut])
async def list_apps(
    session: AsyncSession = Depends(get_async_session),
):
    result = await session.execute(
        select(WorkflowApp).order_by(WorkflowApp.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/api/v1/apps/{app_name}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_app(
    app_name: str,
    session: AsyncSession = Depends(get_async_session),
):
    app_obj = await session.scalar(
        select(WorkflowApp).where(WorkflowApp.name == app_name)
    )
    if not app_obj:
        raise HTTPException(404, f"App '{app_name}' not found")
    await session.delete(app_obj)
    await session.commit()


@router.post("/v1/apps/{app_name}", dependencies=[Depends(require_admin)])
async def execute_app(
    app_name: str,
    body: dict,
    session: AsyncSession = Depends(get_async_session),
):
    """External endpoint: merge user params into snapshot and execute workflow."""
    from src.services.workflow_executor import WorkflowExecutor, ExecutionError
    from src.models.execution_task import ExecutionTask

    app_obj = await session.scalar(
        select(WorkflowApp).where(WorkflowApp.name == app_name)
    )
    if not app_obj:
        raise HTTPException(404, f"App '{app_name}' not found")
    if not app_obj.active:
        raise HTTPException(403, f"App '{app_name}' is inactive")

    snapshot = dict(app_obj.workflow_snapshot)
    nodes = [dict(n) for n in snapshot.get("nodes", [])]
    edges = snapshot.get("edges", [])

    # Merge exposed inputs from request body into the matching node data
    for param in app_obj.exposed_inputs:
        api_name = param["api_name"]
        if api_name in body:
            for node in nodes:
                if node["id"] == param["node_id"]:
                    node.setdefault("data", {})[param["param_key"]] = body[api_name]

    task = ExecutionTask(
        workflow_name=app_obj.name,
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

    # Increment call count
    app_obj.call_count = (app_obj.call_count or 0) + 1
    await session.commit()

    return result
