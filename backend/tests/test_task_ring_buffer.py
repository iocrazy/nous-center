"""Lane B: TaskSnapshot + TaskRingBuffer 单元测试（纯内存，无 DB、无 GPU）。"""
from datetime import datetime, timezone

from src.services.task_ring_buffer import TaskSnapshot


def _make_snapshot(task_id: int = 1, status: str = "queued", **kw) -> TaskSnapshot:
    base = dict(
        task_id=task_id,
        workflow_name="wf",
        status=status,
        priority=10,
        gpu_group=None,
        runner_id=None,
        nodes_total=0,
        nodes_done=0,
        current_node=None,
        queued_at=None,
        started_at=None,
        finished_at=None,
        duration_ms=None,
        error=None,
        cancel_reason=None,
        db_synced=True,
    )
    base.update(kw)
    return TaskSnapshot(**base)


def test_snapshot_construct_defaults():
    snap = _make_snapshot()
    assert snap.task_id == 1
    assert snap.status == "queued"
    assert snap.db_synced is True
    assert snap.gpu_group is None


def test_snapshot_from_orm_task():
    """from_task 把 ExecutionTask ORM 行转成快照，db_synced 由调用方传入。"""

    class _FakeTask:
        # 鸭子类型，避免测试依赖真 ORM/DB
        id = 42
        workflow_name = "laneB-wf"
        status = "running"
        priority = 0
        gpu_group = "image"
        runner_id = "runner-i"
        nodes_total = 3
        nodes_done = 1
        current_node = "sampler"
        queued_at = datetime(2026, 5, 14, tzinfo=timezone.utc)
        started_at = datetime(2026, 5, 14, tzinfo=timezone.utc)
        finished_at = None
        duration_ms = None
        error = None
        cancel_reason = None

    snap = TaskSnapshot.from_task(_FakeTask(), db_synced=False)
    assert snap.task_id == 42
    assert snap.status == "running"
    assert snap.priority == 0
    assert snap.gpu_group == "image"
    assert snap.runner_id == "runner-i"
    assert snap.nodes_done == 1
    assert snap.current_node == "sampler"
    assert snap.db_synced is False
