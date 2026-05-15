"""Lane J integration: queue backlog 503 / server-restart recovery /
DB reconcile / cross-runner tensor serialization roundtrip (spec §5.3
degradation + boundary rows).
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.runner import protocol as P
from src.services.inference.exceptions import QueueFullError
from src.services.scheduler.group_scheduler import (
    PRIORITY_BATCH,
    GroupScheduler,
)
from src.services.task_ring_buffer import TaskRingBuffer, TaskSnapshot

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_queue_backlog_returns_queue_full_error():
    """Queue fills to capacity → next enqueue raises QueueFullError (spec §4.7).

    GroupScheduler is constructed with a small capacity (10) so the test runs
    fast; the default production capacity is QUEUE_CAPACITY=1000. The exception
    carries retry_after_s used by the routing layer to emit Retry-After 503.
    """

    async def _stub_executor(task_id, spec, cancel_event, cancel_flag):
        # never resolves — scheduler will fill the queue while it waits.
        # We don't start() the scheduler in this test, so this body is never
        # called; we only exercise the synchronous capacity check in enqueue.
        return {}

    sched = GroupScheduler(group_id="image", executor=_stub_executor, capacity=10)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Fill to capacity (10 entries).
    for i in range(10):
        await sched.enqueue(
            task_id=i,
            priority=PRIORITY_BATCH,
            queued_at=base + timedelta(seconds=i),
            workflow_spec={},
        )

    with pytest.raises(QueueFullError) as exc_info:
        await sched.enqueue(
            task_id=11,
            priority=PRIORITY_BATCH,
            queued_at=base + timedelta(seconds=11),
            workflow_spec={},
        )
    err = exc_info.value
    assert err.group_id == "image"
    assert err.retry_after_s > 0


@pytest.mark.asyncio
async def test_server_restart_recovery_via_db(db_session):
    """API server restart: status='running' rows are stale on cold boot.

    There is no `recover_tasks_on_startup` function in the codebase yet
    (Lane S deferred the cold-boot scan — the row-level semantics are what
    Lane J asserts here at the DB layer). This test:
      1. Pre-seeds a `running` task and a `queued` task in the DB.
      2. Simulates what a cold-boot recovery routine MUST do: rewrite
         orphan `running` rows to `failed` with reason 'server_restarted'.
      3. Asserts the queued row is untouched (it will simply be re-dispatched
         when the scheduler comes up).

    When Lane S/G adds a real `recover_tasks_on_startup` function, swap the
    inline rewrite below with a call to it — the test's pre/post DB shape
    contract is the regression target.
    """
    from sqlalchemy import select

    from src.models.execution_task import ExecutionTask

    running = ExecutionTask(
        workflow_name="r",
        status="running",
        nodes_total=1,
        nodes_done=0,
    )
    queued = ExecutionTask(
        workflow_name="q",
        status="queued",
        nodes_total=1,
        nodes_done=0,
    )
    db_session.add_all([running, queued])
    await db_session.commit()
    await db_session.refresh(running)
    await db_session.refresh(queued)

    # Simulate cold-boot recovery: rewrite orphan `running` rows.
    rows = (
        await db_session.execute(
            select(ExecutionTask).where(ExecutionTask.status == "running")
        )
    ).scalars().all()
    for row in rows:
        row.status = "failed"
        row.error = "server_restarted"
    await db_session.commit()

    await db_session.refresh(running)
    await db_session.refresh(queued)
    assert running.status == "failed"
    assert running.error == "server_restarted"
    assert queued.status == "queued"


def test_db_reconcile_backfills_unsynced():
    """Ring buffer marks db_synced=False during degradation; reconcile flips them."""
    rb = TaskRingBuffer()
    for i in range(5):
        rb.push(
            TaskSnapshot(
                task_id=i,
                workflow_name="t",
                status="completed",
                priority=PRIORITY_BATCH,
                gpu_group="image",
                runner_id=None,
                nodes_total=1,
                nodes_done=1,
                current_node=None,
                queued_at=None,
                started_at=None,
                finished_at=None,
                duration_ms=1,
                error=None,
                cancel_reason=None,
                db_synced=False,
            )
        )

    unsynced = rb.unsynced()
    assert len(unsynced) == 5

    for snap in unsynced:
        ok = rb.mark_synced(snap.task_id)
        assert ok is True

    assert rb.unsynced() == []
    for snap in rb.list_recent(limit=100):
        assert snap.db_synced is True


def test_cross_runner_tensor_serialization_roundtrip():
    """Main-process view: NodeResult carrying a tensor reference roundtrips
    through the protocol encode/decode (msgpack) intact.

    Real D->H GPU copy is e2e (spec §5.4), out of scope. The integration
    target is the wire envelope: a NodeResult with shape/dtype metadata +
    a storage path is what main-process workflow_executor receives from
    image/TTS runners, then forwards to the next node — the bytes survive
    msgpack roundtrip preserving the nested dict / list types.
    """
    msg = P.NodeResult(
        task_id=1,
        node_id="vae",
        status="completed",
        outputs={
            "path": "outputs/1/latent.bin",
            "meta": {"shape": [1, 4, 128, 128], "dtype": "float16"},
        },
        error=None,
        duration_ms=12,
    )
    raw = P.encode(msg)
    back = P.decode(raw)
    assert back.task_id == msg.task_id
    assert back.status == "completed"
    assert back.outputs["path"] == "outputs/1/latent.bin"
    assert back.outputs["meta"]["shape"] == [1, 4, 128, 128]
    assert back.outputs["meta"]["dtype"] == "float16"
