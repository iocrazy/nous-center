"""Lane G: QueuedTask + GroupScheduler 单元测试（纯内存，无 DB / GPU / runner）。"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.services.scheduler.group_scheduler import (
    PRIORITY_BATCH,
    PRIORITY_INTERACTIVE,
    QueuedTask,
)

_T0 = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _qt(task_id: int, priority: int, offset_s: float = 0.0) -> QueuedTask:
    return QueuedTask.create(
        task_id=task_id,
        priority=priority,
        queued_at=_T0 + timedelta(seconds=offset_s),
        workflow_spec={"id": task_id},
    )


def test_priority_constants():
    assert PRIORITY_INTERACTIVE == 0
    assert PRIORITY_BATCH == 10
    assert PRIORITY_INTERACTIVE < PRIORITY_BATCH


def test_interactive_sorts_before_batch():
    """优先级低的数字先出（interactive=0 在 batch=10 前）。"""
    interactive = _qt(1, PRIORITY_INTERACTIVE, offset_s=100)  # 即使晚入队
    batch = _qt(2, PRIORITY_BATCH, offset_s=0)  # 即使早入队
    assert interactive < batch


def test_same_priority_fifo_by_queued_at():
    """同优先级内，queued_at 早的先出（FIFO）。"""
    early = _qt(1, PRIORITY_BATCH, offset_s=0)
    late = _qt(2, PRIORITY_BATCH, offset_s=5)
    assert early < late


def test_same_priority_same_time_breaks_by_task_id():
    """同优先级 + 同 queued_at（精度内相等）时，task_id 兜底，sort_key 仍可全序。"""
    a = _qt(10, PRIORITY_BATCH, offset_s=0)
    b = _qt(20, PRIORITY_BATCH, offset_s=0)
    assert a < b  # task_id 10 < 20
    # 关键：比较不会因为 workflow_spec dict 不可比而 TypeError
    assert sorted([b, a])[0].task_id == 10


async def test_queued_task_works_in_priority_queue():
    """放进真 asyncio.PriorityQueue，弹出顺序 = 优先级 + FIFO。"""
    q: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()
    await q.put(_qt(1, PRIORITY_BATCH, offset_s=0))
    await q.put(_qt(2, PRIORITY_INTERACTIVE, offset_s=10))  # 晚入队但高优先级
    await q.put(_qt(3, PRIORITY_BATCH, offset_s=1))
    order = []
    while not q.empty():
        order.append((await q.get()).task_id)
    assert order == [2, 1, 3]  # interactive 先, 然后 batch 内 FIFO
