"""Lane B: ExecutionTask V1.5 新列 schema 回归。

新列全部 nullable（priority 例外：有 default=10），旧调用方不传也能 insert。
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from src.models.execution_task import ExecutionTask

pytestmark = pytest.mark.anyio


async def test_v15_columns_default_to_null(db_session):
    """不传 V1.5 列时，insert 成功；priority 落 default=10，其余落 NULL。"""
    task = ExecutionTask(workflow_name="laneB-defaults", status="queued")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    assert task.priority == 10  # default
    assert task.gpu_group is None
    assert task.runner_id is None
    assert task.queued_at is None
    assert task.started_at is None
    assert task.finished_at is None
    assert task.node_timings is None
    assert task.cancel_reason is None


async def test_v15_columns_round_trip(db_session):
    """写入全部 V1.5 列后能原样读回。"""
    now = datetime(2026, 5, 14, 10, 0, 0, tzinfo=timezone.utc)
    task = ExecutionTask(
        workflow_name="laneB-roundtrip",
        status="completed",
        priority=0,
        gpu_group="llm-tp",
        runner_id="runner-i",
        queued_at=now,
        started_at=now,
        finished_at=now,
        node_timings={"node_a": {"duration_ms": 1200, "cached": False}},
        cancel_reason=None,
    )
    db_session.add(task)
    await db_session.commit()
    task_id = task.id

    db_session.expire_all()
    fetched = (
        await db_session.execute(
            select(ExecutionTask).where(ExecutionTask.id == task_id)
        )
    ).scalar_one()

    assert fetched.priority == 0
    assert fetched.gpu_group == "llm-tp"
    assert fetched.runner_id == "runner-i"
    # SQLite + aiosqlite strips tzinfo on round-trip; compare as naive (PG keeps tz).
    naive_now = now.replace(tzinfo=None)
    assert fetched.queued_at.replace(tzinfo=None) == naive_now
    assert fetched.started_at.replace(tzinfo=None) == naive_now
    assert fetched.finished_at.replace(tzinfo=None) == naive_now
    assert fetched.node_timings == {"node_a": {"duration_ms": 1200, "cached": False}}
    assert fetched.cancel_reason is None


async def test_cancel_reason_persists(db_session):
    """cancel_reason 落字符串。"""
    task = ExecutionTask(
        workflow_name="laneB-cancel",
        status="cancelled",
        cancel_reason="user requested at node sampler",
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)
    assert task.cancel_reason == "user requested at node sampler"
