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

from src.models.database import create_session_factory
from src.models.execution_task import ExecutionTask
from src.services.workflow_executor import ExecutionError, WorkflowExecutor

logger = logging.getLogger(__name__)


async def _broadcast(channel_id: str | None, event: dict) -> None:
    """推 WS 进度事件 —— 复用现有 /ws/workflow/{channel_id} 连接桶。"""
    if not channel_id:
        return
    from src.api.main import _ws_connections

    for ws in list(_ws_connections.get(channel_id, [])):
        try:
            await ws.send_json(event)
        except Exception as e:  # noqa: BLE001 — WS 推送失败静默吞（spec §4.1）
            logger.warning("workflow_runner broadcast failed: %s", e)


async def run_workflow_task(
    task_id: int,
    workflow_data: dict,
    runner_client: Any = None,
    channel_id: str | None = None,
) -> None:
    """后台执行一个 workflow，把终态写回 ExecutionTask。

    本函数不抛出 —— 所有异常都落到 task.status=failed + task.error。调用方
    （create_task）拿不到也不该拿返回值。
    """
    start = time.monotonic()
    nodes = workflow_data.get("nodes", [])

    async def on_progress(event: dict) -> None:
        await _broadcast(channel_id, event)

    executor = WorkflowExecutor(
        workflow_data,
        on_progress=on_progress if channel_id else None,
        runner_client=runner_client,
    )

    session_factory = create_session_factory()
    async with session_factory() as session:
        task = await session.get(ExecutionTask, task_id)
        if task is None:
            logger.error("run_workflow_task: task %s not found", task_id)
            return
        task.status = "running"
        await session.commit()

        try:
            result = await executor.execute()
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "completed"
            task.result = result
            task.duration_ms = elapsed
            task.nodes_done = len(nodes)
            task.current_node = None
            await session.commit()
            await _broadcast(channel_id, {"type": "complete", "progress": 100})
        except ExecutionError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "failed"
            task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            logger.error("workflow %s failed: %s", task_id, e)
        except Exception as e:  # noqa: BLE001 — 后台 task 永不冒泡
            elapsed = int((time.monotonic() - start) * 1000)
            task.status = "failed"
            task.error = str(e)
            task.duration_ms = elapsed
            await session.commit()
            logger.error("workflow %s errored: %s", task_id, e, exc_info=True)
