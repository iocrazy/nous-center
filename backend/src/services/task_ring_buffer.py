"""TaskRingBuffer —— 主进程内最近 200 条 task 快照的热缓存。

DB（execution_tasks 表）是真相源（survives restart）；ring buffer 是热缓存
（survives request burst，O(1) 读，给 TaskPanel / Dashboard 用）。

每条快照带 db_synced 标志：DB 写成功 → True；DB 不可达降级期写失败 → False。
该标志驱动 spec §4.6 的 reconcile（DB 恢复后批量补写 db_synced=False 的条目）。
**本 Lane（B）只提供数据结构与 db_synced 翻转 API；reconcile loop 由后续 Lane 实现。**
"""
from __future__ import annotations

import collections  # noqa: F401 — used by TaskRingBuffer (Task 4)
from dataclasses import dataclass, replace  # noqa: F401 — replace used by TaskRingBuffer (Task 4)
from datetime import datetime
from typing import Any

RING_CAPACITY = 200


@dataclass
class TaskSnapshot:
    """execution_tasks 一行的可观测子集 + db_synced 标志。

    字段对齐 ExecutionTask ORM（spec §3.1）的「前端 / 调度需要看」的子集；
    不含 result / node_timings 这类大载荷（TaskPanel 按需单独查 DB）。
    """

    task_id: int
    workflow_name: str
    status: str  # queued/running/completed/failed/cancelled
    priority: int
    gpu_group: str | None
    runner_id: str | None
    nodes_total: int
    nodes_done: int
    current_node: str | None
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    error: str | None
    cancel_reason: str | None
    db_synced: bool = True

    @classmethod
    def from_task(cls, task: Any, *, db_synced: bool) -> "TaskSnapshot":
        """从 ExecutionTask ORM 行（或鸭子类型等价物）构造快照。

        db_synced 必须显式传入 —— 它反映的是「这次 DB 写有没有成功」，
        是调用方（scheduler / executor）才知道的事，不是 task 行本身的属性。
        """
        return cls(
            task_id=task.id,
            workflow_name=task.workflow_name,
            status=task.status,
            priority=task.priority,
            gpu_group=task.gpu_group,
            runner_id=task.runner_id,
            nodes_total=task.nodes_total,
            nodes_done=task.nodes_done,
            current_node=task.current_node,
            queued_at=task.queued_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            duration_ms=task.duration_ms,
            error=task.error,
            cancel_reason=task.cancel_reason,
            db_synced=db_synced,
        )
