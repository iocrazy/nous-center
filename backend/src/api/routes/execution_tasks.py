import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.execution_task import ExecutionTask
from src.utils.constants import VALID_TASK_STATUSES
from src.api.websocket import ws_manager

logger = logging.getLogger(__name__)

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
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """对齐 ComfyUI /interrupt 行为:running 任务真正中止(不只是改 DB)。
    桥接 HTTP cancel → RunnerClient.abort 给所有 runner(image/tts);runner 内置 cancel_flag 通路
    + 真 adapter 的 callback_on_step_end check → step 边界 raise CancelledError → 落 cancelled
    NodeResult。修了「点 cancel 但 GPU kernel 还跑完整轮」的真 bug。"""
    task = await session.get(ExecutionTask, task_id)
    if not task:
        raise HTTPException(404)
    if task.status not in ("queued", "running"):
        raise HTTPException(400, "Can only cancel queued or running tasks")
    # 向 runners 发 Abort(广播给所有 group;不知道任务在哪个 runner 上,每个 runner 自己
    # 据 task_id 决定;没有该任务的 runner 收到也只是设个 flag 然后被 pop,无副作用)。
    runner_clients = getattr(request.app.state, "runner_clients", None) or {}
    for group_id, client in runner_clients.items():
        try:
            await client.abort(task_id)
        except Exception as e:  # noqa: BLE001 —— 单 runner 失败不阻断其它 / 不阻断 DB 落态
            logger.warning("cancel_task: runner %s abort(%d) failed: %s", group_id, task_id, e)
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


def _detect_llm_meta(result: object) -> dict:
    """PR-1c:LLM workflow result envelope 识别 —— `text` 字符串 + `usage` dict 是
    LLMNode.invoke/stream 的标准返回形状(见 src/services/nodes/llm.py)。
    返 completion_tokens / prompt_tokens 供前端 callout 显示「47 tokens · 23 tok/s」。"""
    out: dict = {
        "task_type": None,
        "llm_prompt_tokens": None,
        "llm_completion_tokens": None,
    }
    if not isinstance(result, dict):
        return out
    for v in result.values():
        if not isinstance(v, dict):
            continue
        # LLM 节点返回 {text: str, usage: {...}, duration_ms: int}
        if isinstance(v.get("text"), str) and isinstance(v.get("usage"), dict):
            out["task_type"] = "llm"
            usage = v["usage"]
            out["llm_prompt_tokens"] = usage.get("prompt_tokens")
            out["llm_completion_tokens"] = usage.get("completion_tokens")
            return out
    return out


def _detect_tts_meta(result: object) -> dict:
    """PR-1b:TTS workflow result envelope 识别 —— audio/* media_type 或 audio_url。
    匹配 ImageBackend 落 image_output_storage 后的形状的 TTS 对应版本(audio_url + duration)。
    返 duration_seconds 供前端音频时长展示。"""
    out: dict = {"task_type": None, "audio_duration_seconds": None}
    if not isinstance(result, dict):
        return out
    for v in result.values():
        if not isinstance(v, dict):
            continue
        media_type = v.get("media_type")
        is_audio = (
            (isinstance(media_type, str) and media_type.startswith("audio/"))
            or "audio_url" in v
        )
        if is_audio:
            out["task_type"] = "tts"
            dur = v.get("duration_seconds") or (v.get("meta") or {}).get("duration_seconds")
            if isinstance(dur, (int, float)):
                out["audio_duration_seconds"] = float(dur)
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
    # 顺序:image 先(workflow 多含 image_output);未命中再 TTS、再 LLM 检测(LLM envelope
    # 是 {text, usage} 最普遍,放最后避免误识别 — TTS / image 都不会同时有 text+usage 形状)。
    # vision 由 PR-1d 扩展(_detect_vision_meta)。
    img_meta = _detect_image_meta(t.result)
    d.update(img_meta)
    if img_meta.get("task_type") is None:
        tts_meta = _detect_tts_meta(t.result)
        if tts_meta.get("task_type"):
            d["task_type"] = tts_meta["task_type"]
            d["audio_duration_seconds"] = tts_meta["audio_duration_seconds"]
        else:
            llm_meta = _detect_llm_meta(t.result)
            if llm_meta.get("task_type"):
                d["task_type"] = llm_meta["task_type"]
                d["llm_prompt_tokens"] = llm_meta["llm_prompt_tokens"]
                d["llm_completion_tokens"] = llm_meta["llm_completion_tokens"]
    # PR-1a/1b/1c(2026-05-27 任务面板重置 spec §State model):显式 `type` 字段
    # (image / tts / llm / vision),对应前端 ServiceType。type=None → 旧 fake / 未识别
    # workflow,前端 Other 兜底。
    d["type"] = d.get("task_type")
    return d
