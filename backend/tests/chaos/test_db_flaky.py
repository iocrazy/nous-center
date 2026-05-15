"""Lane J chaos: 50% DB OperationalError, 1000-task soak →
ring buffer + reconcile reach final consistency.

spec §5.5 test_db_flaky. Manual weekly run: pytest -m chaos.

Uses an in-memory DB stand-in + deterministic RNG so the test exercises
the "ring buffer falls back when DB writes fail, reconcile flips db_synced
once DB recovers" final-consistency property without flaky real-DB
monkeypatching. When Lane G/S adds a real reconcile loop, this can be
upgraded to monkeypatch session.commit and exercise the wire path.
"""
import random

import pytest

from src.services.task_ring_buffer import (
    RING_CAPACITY,
    TaskRingBuffer,
    TaskSnapshot,
)

pytestmark = pytest.mark.chaos


def _snap(tid: int, *, db_synced: bool, priority: int = 10) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=tid,
        workflow_name="soak",
        status="completed",
        priority=priority,
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
        db_synced=db_synced,
    )


@pytest.mark.asyncio
async def test_db_intermittent_failures():
    """50% DB write failure, 1000 tasks:

    - Each task always lands in the ring buffer (in-memory).
    - DB write succeeds 50% of the time → that snapshot gets db_synced=True.
    - The remaining snapshots are db_synced=False.
    - Reconcile (run after "DB recovery"): walk unsynced, write to DB,
      flip the flag.
    - Final consistency: all snapshots in the ring buffer must have
      db_synced=True, and DB's record of each surviving task must match
      the ring buffer's status.
    """
    rng = random.Random(42)  # fixed seed → reproducible chaos
    rb = TaskRingBuffer()
    db_rows: dict[int, str] = {}

    def _try_db_write(task_id: int, status: str) -> bool:
        if rng.random() < 0.5:
            return False  # simulate OperationalError
        db_rows[task_id] = status
        return True

    for tid in range(1000):
        synced = _try_db_write(tid, "completed")
        rb.push(_snap(tid, db_synced=synced, priority=rng.choice([0, 10])))

    # Mid-soak state: ring buffer is bounded by RING_CAPACITY, with a mix.
    recent = rb.list_recent(limit=10_000)
    assert len(recent) <= RING_CAPACITY
    unsynced_before = [s for s in recent if not s.db_synced]
    assert unsynced_before, "50% failure rate should leave some unsynced rows"

    # Reconcile: DB has recovered → write each unsynced row + flip flag.
    for snap in unsynced_before:
        db_rows[snap.task_id] = snap.status
        flipped = rb.mark_synced(snap.task_id)
        assert flipped is True

    # Final consistency: ring buffer fully synced + DB matches.
    recent_after = rb.list_recent(limit=10_000)
    assert all(s.db_synced for s in recent_after), (
        "after reconcile every snapshot must be db_synced"
    )
    for snap in recent_after:
        assert db_rows.get(snap.task_id) == snap.status, (
            f"DB/ring buffer disagree on task {snap.task_id}"
        )


@pytest.mark.asyncio
async def test_db_flaky_ring_buffer_never_loses_recent():
    """DB unreachable for the whole soak; ring buffer still retains the
    most-recent RING_CAPACITY snapshots (degradation does not drop hot data).
    """
    rb = TaskRingBuffer()
    for tid in range(500):
        rb.push(_snap(tid, db_synced=False))

    recent = rb.list_recent(limit=10_000)
    assert len(recent) == RING_CAPACITY
    ids = {s.task_id for s in recent}
    # The most-recent RING_CAPACITY ids are 500-RING_CAPACITY .. 499.
    assert ids == set(range(500 - RING_CAPACITY, 500))
