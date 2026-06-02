"""后台 workflow 执行入口 —— D17 纯异步契约的执行侧。

两个 /run 端点（instance_run / execute_workflow_direct）建完 ExecutionTask 后，
用 asyncio.create_task(run_workflow_task(...)) 把执行丢到后台，立即返回 202。
本函数负责：建 WorkflowExecutor、跑、把终态写回 ExecutionTask、推 WS complete。

注意：本函数自己开一个独立 DB session —— 它跑在 request 之外的后台 task 里，
不能复用 request-scoped 的 session（那个在 handler 返回时就关了）。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.models.database import get_session_factory
from src.models.execution_task import ExecutionTask
from src.services.prediction_service import fire_webhook, task_to_prediction
from src.services.workflow_executor import ExecutionError, WorkflowExecutor

logger = logging.getLogger(__name__)


async def _webhook(task: ExecutionTask, event: str) -> None:
    """PR-3:在 start/终态点 POST Prediction 到 task.webhook_url(best-effort,有就发)。"""
    url = getattr(task, "webhook_url", None)
    if not url:
        return
    pred = task_to_prediction(task, service=task.workflow_name)
    await fire_webhook(url, getattr(task, "webhook_events", None), event, pred)


async def _broadcast(channel_id: str | None, event: dict) -> None:
    """推 WS 进度事件 —— 复用现有 /ws/workflow/{channel_id} 连接桶。"""
    if not channel_id:
        return
    from src.api.main import _ws_connections

    sockets = _ws_connections.get(channel_id)
    if not sockets:
        return
    # 非干净断开(网络 reset / 进程杀)的客户端不抛 WebSocketDisconnect,endpoint 的 remove
    # 不执行 → 死连接永久驻留,每次广播重试 + 刷 warning(round2 #6)。send 失败即剔除,
    # 桶空删 key,防连接对象 + channel_id 双重泄漏。
    dead = []
    for ws in list(sockets):
        try:
            await ws.send_json(event)
        except Exception as e:  # noqa: BLE001 — WS 推送失败静默吞（spec §4.1）
            logger.warning("workflow_runner broadcast failed,剔除死连接: %s", e)
            dead.append(ws)
    for ws in dead:
        try:
            sockets.remove(ws)
        except ValueError:
            pass
    if not sockets:
        _ws_connections.pop(channel_id, None)


async def _broadcast_task_status(task: ExecutionTask, event: str = "updated") -> None:
    """PR-5(2026-05-28 任务面板重置 WS 实时更新):task 状态变化时广播全局 `/ws/tasks`。

    `workflow_runner` 是后台 task,跟 routes/execution_tasks.py 的 record/cancel/delete
    路径平行 —— 那 3 个 endpoint 已在 status 变化时调 broadcast_task_update,
    但 **run 路径(running/completed/failed)以前没调**,导致前端 useTasks 只能等 60s
    polling fallback 收到新状态(UX 表现为「点 Run 后等好久任务才出现」+ 「task 跑完
    UI 隔几十秒才更新」)。

    复用 routes 的 `_task_to_dict` 序列化(含 PR-1a/b/c/d 加的 type / audio_duration_seconds
    / llm_*_tokens / vision_completion_tokens 字段),保持 WS payload 跟 REST 一致。
    WS 推送失败静默吞(spec §4.1)。
    """
    try:
        from src.api.routes.execution_tasks import _task_to_dict
        from src.api.websocket import ws_manager
        await ws_manager.broadcast_task_update(event, _task_to_dict(task))
    except Exception as e:  # noqa: BLE001
        logger.warning("broadcast_task_status failed: %s", e)


async def run_workflow_task(
    task_id: int,
    workflow_data: dict,
    runner_client: Any = None,
    runner_clients: dict | None = None,
    channel_id: str | None = None,
) -> None:
    """后台执行一个 workflow，把终态写回 ExecutionTask。

    本函数不抛出 —— 所有异常都落到 task.status=failed + task.error。调用方
    （create_task）拿不到也不该拿返回值。

    Lane K: runner_clients (dict) 是多 group 入口,routes 从 app.state 取后
    传进来。runner_client (单数) 为兼容旧测试 / inline-only 老路径保留。
    """
    start = time.monotonic()
    nodes = workflow_data.get("nodes", [])

    async def on_progress(event: dict) -> None:
        await _broadcast(channel_id, event)

    # 先取 workflow_name 给 executor → RunnerClient.run_node 用做 current_task 显示。
    session_factory = get_session_factory()
    async with session_factory() as session:
        task = await session.get(ExecutionTask, task_id)
        if task is None:
            logger.error("run_workflow_task: task %s not found", task_id)
            return
        task.status = "running"
        await session.commit()
        wf_name = task.workflow_name or ""
        # PR-5:广播 running 状态到 /ws/tasks(前端 useTasks 立即翻新)
        await _broadcast_task_status(task, event="updated")
        await _webhook(task, "start")  # PR-3

    executor = WorkflowExecutor(
        workflow_data,
        on_progress=on_progress if channel_id else None,
        runner_client=runner_client,
        runner_clients=runner_clients,
        task_id=task_id,
        workflow_name=wf_name,
    )

    async with session_factory() as session:
        task = await session.get(ExecutionTask, task_id)
        if task is None:
            logger.error("run_workflow_task: task %s vanished mid-exec", task_id)
            return

        try:
            result = await executor.execute()
            elapsed = int((time.monotonic() - start) * 1000)
            # round2 #cancel-race:HTTP cancel 路径可能在 execute() 期间把 status 写成
            # cancelled。executor 没感知取消而正常跑完时,success 路径若直接覆盖成
            # completed 会丢掉用户的取消意图(跟下面 except 路径的 PR-3 guard 对称)。
            await session.refresh(task)
            if task.status == "cancelled":
                task.duration_ms = elapsed
                await session.commit()
                logger.info("workflow %s completed after cancel — honoring cancelled", task_id)
                await _broadcast_task_status(task, event="updated")
                await _webhook(task, "completed")  # PR-3
                return
            task.status = "completed"
            task.result = result
            task.duration_ms = elapsed
            task.nodes_done = len(nodes)
            task.current_node = None
            await session.commit()
            await _broadcast(channel_id, {"type": "complete", "progress": 100})
            # PR-5:广播 completed 状态到 /ws/tasks
            await _broadcast_task_status(task, event="updated")
            await _webhook(task, "completed")  # PR-3
        except ExecutionError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            # PR-3:HTTP cancel 在另一路径已把 status 写 cancelled —— 不要覆盖成 failed。
            await session.refresh(task)
            if task.status != "cancelled":
                task.status = "failed"
                task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            await _broadcast_task_status(task, event="updated")  # PR-5
            await _webhook(task, "completed")  # PR-3(终态含 failed)
            # 用 ERROR 级别 + exc_info=True 输出完整 traceback —— ExecutionError 是顶层包装,
            # 真实节点错误(RuntimeError/CUDA OOM 等)在 __cause__ 链上,需要 traceback 才能看见。
            logger.error(
                "workflow %s end: status=%s err=%s",
                task_id, task.status, e,
                exc_info=task.status != "cancelled",
            )
        except Exception as e:  # noqa: BLE001 — 后台 task 永不冒泡
            elapsed = int((time.monotonic() - start) * 1000)
            await session.refresh(task)
            if task.status != "cancelled":
                task.status = "failed"
                task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            await _broadcast_task_status(task, event="updated")  # PR-5
            await _webhook(task, "completed")  # PR-3(终态含 failed)
            logger.info("workflow %s end: status=%s err=%s", task_id, task.status, e, exc_info=task.status != "cancelled")
