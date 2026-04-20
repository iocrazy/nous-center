import asyncio
from src.services.task_queue import TaskQueue


async def _echo(params):
    return params.get("value", "ok")


async def _slow(params):
    await asyncio.sleep(5)
    return "done"


async def _failing(params):
    raise RuntimeError("boom")


async def test_submit_and_complete():
    q = TaskQueue(max_concurrent=2, default_timeout=10)
    task_id = await q.submit(_echo, {"value": "hello"})
    await asyncio.sleep(0.2)
    status = q.get_status(task_id)
    assert status["status"] == "completed"
    assert status["result"] == "hello"


async def test_timeout():
    q = TaskQueue(max_concurrent=2, default_timeout=10)
    task_id = await q.submit(_slow, {}, timeout=0.1)
    await asyncio.sleep(0.5)
    status = q.get_status(task_id)
    assert status["status"] == "timeout"


async def test_cancel():
    q = TaskQueue(max_concurrent=1, default_timeout=10)
    await q.submit(_slow, {}, timeout=10)
    await asyncio.sleep(0.05)
    task_id = await q.submit(_echo, {"value": "x"})
    cancelled = await q.cancel(task_id)
    assert cancelled
    status = q.get_status(task_id)
    assert status["status"] == "cancelled"


async def test_retry():
    call_count = 0

    async def fail_then_succeed(params):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise MemoryError("CUDA OOM")
        return "ok"

    q = TaskQueue(max_concurrent=2, default_timeout=10)
    task_id = await q.submit(fail_then_succeed, {}, max_retries=2)
    await asyncio.sleep(3)
    status = q.get_status(task_id)
    assert status["status"] == "completed"
    assert status["result"] == "ok"


async def test_non_retryable_error():
    q = TaskQueue(max_concurrent=2, default_timeout=10)
    task_id = await q.submit(_failing, {}, max_retries=2)
    await asyncio.sleep(0.3)
    status = q.get_status(task_id)
    assert status["status"] == "failed"
    assert "boom" in status["error"]


async def test_concurrency_limit():
    running = 0
    max_running = 0

    async def track(params):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0.1)
        running -= 1

    q = TaskQueue(max_concurrent=2, default_timeout=10)
    ids = []
    for _ in range(5):
        ids.append(await q.submit(track, {}))
    await asyncio.sleep(1.5)
    assert max_running <= 2
