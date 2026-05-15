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


from src.services.task_ring_buffer import RING_CAPACITY, TaskRingBuffer


def test_push_and_get():
    rb = TaskRingBuffer()
    snap = _make_snapshot(task_id=100)
    rb.push(snap)
    assert rb.get(100) is snap
    assert rb.get(999) is None
    assert len(rb) == 1


def test_update_in_place_no_duplicate():
    """同 task_id 第二次 push 应替换，不产生重复条目。"""
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, status="queued"))
    rb.push(_make_snapshot(task_id=1, status="running", nodes_done=2))
    assert len(rb) == 1
    assert rb.get(1).status == "running"
    assert rb.get(1).nodes_done == 2
    # list_recent 里也只有一条
    assert [s.task_id for s in rb.list_recent()] == [1]


def test_maxlen_evicts_oldest():
    """超过 RING_CAPACITY 后，最旧的被 evict，_by_id 同步清理。"""
    rb = TaskRingBuffer()
    for i in range(RING_CAPACITY + 5):
        rb.push(_make_snapshot(task_id=i))
    assert len(rb) == RING_CAPACITY
    # task_id 0..4 应被 evict
    for evicted in range(5):
        assert rb.get(evicted) is None
    # task_id 5..204 应还在
    assert rb.get(5) is not None
    assert rb.get(RING_CAPACITY + 4) is not None


def test_list_recent_order_and_limit():
    """list_recent 返回最近优先（新 → 旧），limit 截断。"""
    rb = TaskRingBuffer()
    for i in range(10):
        rb.push(_make_snapshot(task_id=i))
    recent = rb.list_recent(limit=3)
    assert [s.task_id for s in recent] == [9, 8, 7]
    # 不传 limit 返回全部（最近优先）
    assert [s.task_id for s in rb.list_recent()] == list(range(9, -1, -1))


def test_mark_synced_flips_flag():
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, db_synced=False))
    assert rb.get(1).db_synced is False
    ok = rb.mark_synced(1)
    assert ok is True
    assert rb.get(1).db_synced is True
    # 不存在的 id 返回 False
    assert rb.mark_synced(999) is False


def test_unsynced_lists_only_false():
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, db_synced=True))
    rb.push(_make_snapshot(task_id=2, db_synced=False))
    rb.push(_make_snapshot(task_id=3, db_synced=False))
    unsynced_ids = sorted(s.task_id for s in rb.unsynced())
    assert unsynced_ids == [2, 3]


def test_update_in_place_can_change_db_synced():
    """降级期 push(db_synced=False)，DB 恢复后 push(db_synced=True) 覆盖。"""
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, status="running", db_synced=False))
    assert rb.get(1).db_synced is False
    rb.push(_make_snapshot(task_id=1, status="completed", db_synced=True))
    assert rb.get(1).db_synced is True
    assert rb.get(1).status == "completed"
    assert rb.unsynced() == []
