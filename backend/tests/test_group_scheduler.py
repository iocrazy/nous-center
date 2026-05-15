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


# ---------------------------------------------------------------------------
# GroupScheduler tests (Task 5)
# ---------------------------------------------------------------------------

from src.services.inference.exceptions import NodeCancelled, QueueFullError  # noqa: E402
from src.services.scheduler.group_scheduler import (  # noqa: E402
    QUEUE_CAPACITY,
    GroupScheduler,
)


def _make_scheduler(executor):
    return GroupScheduler(group_id="image", executor=executor)


async def test_enqueue_dispatch_completes():
    """enqueue → dispatcher 弹出 → executor 跑 → task 进 results。"""
    results = {}

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        results[task_id] = {"ok": True, "spec": spec}
        return results[task_id]

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(
        task_id=1, priority=PRIORITY_INTERACTIVE,
        queued_at=_T0, workflow_spec={"id": 1},
    )
    await sched.join()  # 等队列里的 task 全部派发完
    await sched.stop()
    assert results == {1: {"ok": True, "spec": {"id": 1}}}


async def test_priority_order_interactive_first():
    """batch task 先入队，interactive 后入队 —— interactive 先被 executor 跑。"""
    run_order = []

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        run_order.append(task_id)
        return {}

    sched = _make_scheduler(fake_executor)
    # 先把两个 task 灌进队列再 start —— 保证 dispatcher 启动时队列里已有 2 个，
    # 排序才有意义（否则先 enqueue 的可能在第二个 enqueue 前就被弹走了）。
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.enqueue(task_id=2, priority=PRIORITY_INTERACTIVE,
                        queued_at=_T0, workflow_spec={})
    await sched.start()
    await sched.join()
    await sched.stop()
    assert run_order == [2, 1]  # interactive 先


async def test_cancel_while_queued_skips_executor():
    """task 还在排队（未 dispatch）时 cancel —— dispatcher 弹出时跳过，
    executor 根本不被调用，task 标 cancelled。"""
    executor_calls = []

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        executor_calls.append(task_id)
        return {}

    sched = _make_scheduler(fake_executor)
    # 不 start —— 先 enqueue + cancel，再 start，保证 cancel 发生在 dispatch 前
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    sched.request_cancel(1, reason="user changed mind")
    await sched.start()
    await sched.join()
    await sched.stop()
    assert executor_calls == []  # executor 从未被调
    assert sched.get_status(1) == "cancelled"


async def test_cancel_inflight_sets_cancel_flag():
    """task 正在执行时 cancel —— cancel_event 和 CancelFlag 都被 set，
    executor 能观察到 → 抛 NodeCancelled → task 标 cancelled。"""
    started = asyncio.Event()

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        started.set()
        # 模拟一个长节点：轮询 cancel_flag（adapter 实际是 callback 里 check）
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cancel_flag.is_set():
                raise NodeCancelled(cancel_flag.reason or "cancelled")
        return {}

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_INTERACTIVE,
                        queued_at=_T0, workflow_spec={})
    await started.wait()  # 等 executor 真的开始跑
    sched.request_cancel(1, reason="abort inflight")
    await sched.join()
    await sched.stop()
    assert sched.get_status(1) == "cancelled"
    assert sched.get_cancel_reason(1) == "abort inflight"


async def test_executor_exception_marks_failed():
    """executor 抛非 cancel 异常 —— task 标 failed，dispatcher 不挂、继续服务。"""
    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        raise RuntimeError("node OOM")

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    # dispatcher 仍活着 —— 再灌一个能成功的
    ok = {}

    async def ok_executor(task_id, spec, cancel_event, cancel_flag):
        ok[task_id] = True
        return {}

    sched._executor = ok_executor  # 换 executor 验证 loop 没死
    await sched.enqueue(task_id=2, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    await sched.stop()
    assert sched.get_status(1) == "failed"
    assert sched.get_status(2) == "completed"


async def test_enqueue_raises_queue_full_on_backlog():
    """队列堆积到 QUEUE_CAPACITY 后，enqueue 抛 QueueFullError。"""
    async def slow_executor(task_id, spec, cancel_event, cancel_flag):
        await asyncio.sleep(60)  # 永远跑不完 —— 让队列堆起来
        return {}

    sched = _make_scheduler(slow_executor)
    # 不 start dispatcher —— 队列只进不出，直接灌满
    for i in range(QUEUE_CAPACITY):
        await sched.enqueue(task_id=i, priority=PRIORITY_BATCH,
                            queued_at=_T0, workflow_spec={})
    with pytest.raises(QueueFullError) as ei:
        await sched.enqueue(task_id=99999, priority=PRIORITY_BATCH,
                            queued_at=_T0, workflow_spec={})
    assert ei.value.group_id == "image"
    assert ei.value.capacity == QUEUE_CAPACITY


async def test_inflight_cleared_after_completion():
    """task 终态后从 inflight_tasks / cancel_events 清理，不泄漏。"""
    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        return {}

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    await sched.stop()
    assert 1 not in sched.inflight_tasks
    assert 1 not in sched.cancel_events
