from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.execution_task import ExecutionTask
from src.utils.constants import VALID_TASK_STATUSES
from src.api.websocket import ws_manager

router = APIRouter(prefix="/api/v1/tasks", tags=["execution-tasks"])


@router.get("")
async def list_tasks(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        select(ExecutionTask)
        .order_by(desc(ExecutionTask.created_at))
        .limit(limit)
        .offset(offset)
    )
    if status:
        stmt = stmt.where(ExecutionTask.status == status)
    result = await session.execute(stmt)
    tasks = result.scalars().all()
    return [_task_to_dict(t) for t in tasks]


@router.post("/record")
async def record_task(
    body: dict,
    session: AsyncSession = Depends(get_async_session),
):
    """Record a task from frontend execution."""
    status = body.get("status", "completed")
    if status not in VALID_TASK_STATUSES:
        raise HTTPException(400, f"Invalid status: {status}")

    task = ExecutionTask(
        workflow_name=str(body.get("workflow_name", ""))[:100],
        status=status,
        nodes_total=int(body.get("nodes_total", 0)),
        nodes_done=int(body.get("nodes_done", 0)),
        error=str(body["error"])[:2000] if body.get("error") else None,
        duration_ms=int(body["duration_ms"]) if body.get("duration_ms") is not None else None,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    result = _task_to_dict(task)
    await ws_manager.broadcast_task_update("created", result)
    return result


@router.get("/{task_id}")
async def get_task(
    task_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    task = await session.get(ExecutionTask, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return _task_to_dict(task)


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    task = await session.get(ExecutionTask, task_id)
    if not task:
        raise HTTPException(404)
    if task.status not in ("queued", "running"):
        raise HTTPException(400, "Can only cancel queued or running tasks")
    task.status = "cancelled"
    await session.commit()
    await ws_manager.broadcast_task_update("updated", _task_to_dict(task))
    return {"status": "cancelled"}


@router.post("/{task_id}/retry")
async def retry_task(
    task_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    task = await session.get(ExecutionTask, task_id)
    if not task or task.status not in ("failed", "cancelled"):
        raise HTTPException(400, "Can only retry failed or cancelled tasks")
    # Create a new task with same workflow
    new_task = ExecutionTask(
        workflow_id=task.workflow_id,
        workflow_name=task.workflow_name,
        status="queued",
        nodes_total=task.nodes_total,
    )
    session.add(new_task)
    await session.commit()
    await session.refresh(new_task)
    return _task_to_dict(new_task)


@router.delete("/{task_id}")
async def delete_task(
    task_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    task = await session.get(ExecutionTask, task_id)
    if not task:
        raise HTTPException(404)
    task_dict = _task_to_dict(task)
    await session.delete(task)
    await session.commit()
    await ws_manager.broadcast_task_update("deleted", task_dict)
    return {"status": "deleted"}


def _detect_image_meta(result: object) -> dict:
    """Pluck task_type + size from a workflow result by scanning for the
    image_output envelope shape. Stays None for non-image results so the
    UI can skip the badge entirely.
    """
    out: dict = {"task_type": None, "image_width": None, "image_height": None}
    if not isinstance(result, dict):
        return out
    for v in result.values():
        if not isinstance(v, dict):
            continue
        media_type = v.get("media_type")
        is_image = (
            (isinstance(media_type, str) and media_type.startswith("image/"))
            or "image_url" in v
        )
        if is_image:
            out["task_type"] = "image"
            w, h = v.get("width"), v.get("height")
            if isinstance(w, int) and isinstance(h, int):
                out["image_width"] = w
                out["image_height"] = h
            return out
    return out


def _task_to_dict(t: ExecutionTask) -> dict:
    d = {
        "id": str(t.id),
        "workflow_id": str(t.workflow_id) if t.workflow_id else None,
        "workflow_name": t.workflow_name,
        "status": t.status,
        "nodes_total": t.nodes_total,
        "nodes_done": t.nodes_done,
        "current_node": t.current_node,
        "result": t.result,
        "error": t.error,
        "duration_ms": t.duration_ms,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }
    d.update(_detect_image_meta(t.result))
    return d
