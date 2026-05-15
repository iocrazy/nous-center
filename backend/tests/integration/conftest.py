"""Lane J integration suite shared fixtures.

fake_runner / fake_vllm / hardware_topo are registered globally via the
top-level tests/conftest.py pytest_plugins block — this file only adds
integration-specific composites.
"""
import pytest

from src.services.scheduler.group_scheduler import GroupScheduler
from src.services.task_ring_buffer import TaskRingBuffer


class _SchedEnv:
    """Minimal scheduling environment for integration tests."""

    def __init__(self, runner, scheduler, ring_buffer) -> None:
        self.runner = runner
        self.scheduler = scheduler
        self.ring_buffer = ring_buffer


@pytest.fixture
def scheduler_env(fake_runner):
    """One image fake_runner + a fresh GroupScheduler + a fresh TaskRingBuffer.

    GroupScheduler takes an executor callable in its constructor; tests that
    want a real dispatch loop should construct their own GroupScheduler with
    a wired executor. This fixture's scheduler is a "shell" with a no-op
    executor — most integration cases drive the runner directly via the
    RunnerClient and only use the ring buffer / scheduler for state assertions.
    """

    async def _noop_executor(task_id, spec, cancel_event, cancel_flag):
        return {}

    runner = fake_runner(group_id="image", gpus=[2])
    scheduler = GroupScheduler(group_id="image", executor=_noop_executor)
    ring_buffer = TaskRingBuffer()
    return _SchedEnv(runner=runner, scheduler=scheduler, ring_buffer=ring_buffer)
