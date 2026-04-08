"""Async task queue with priority, concurrency limits, timeout, and retry."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger(__name__)

# Errors that can be retried (transient)
_RETRYABLE_ERRORS = (MemoryError, OSError, ConnectionError)


@dataclass
class _TaskEntry:
    task_id: str
    func: Callable[..., Coroutine]
    params: dict
    priority: int  # higher = more urgent
    timeout: float
    max_retries: int
    retries: int = 0
    status: str = "queued"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _cancelled: bool = False

    def __lt__(self, other: _TaskEntry) -> bool:
        """Higher priority first, then earlier creation."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class TaskQueue:
    def __init__(self, max_concurrent: int = 4, default_timeout: int = 300):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._default_timeout = default_timeout
        self._tasks: dict[str, _TaskEntry] = {}
        self._queue: asyncio.PriorityQueue[_TaskEntry] = asyncio.PriorityQueue()
        self._worker_task: asyncio.Task | None = None
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if loop and (self._worker_task is None or self._worker_task.done()):
            self._worker_task = loop.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        while True:
            entry = await self._queue.get()
            if entry._cancelled:
                self._queue.task_done()
                continue
            asyncio.create_task(self._run_task(entry))
            self._queue.task_done()

    async def _run_task(self, entry: _TaskEntry) -> None:
        async with self._semaphore:
            if entry._cancelled:
                return

            entry.status = "running"
            entry.started_at = time.time()

            try:
                result = await asyncio.wait_for(
                    entry.func(entry.params),
                    timeout=entry.timeout,
                )
                entry.status = "completed"
                entry.result = result
            except asyncio.TimeoutError:
                entry.status = "timeout"
                entry.error = f"Timed out after {entry.timeout}s"
                logger.warning("Task %s timed out", entry.task_id)
            except Exception as e:
                if isinstance(e, _RETRYABLE_ERRORS) and entry.retries < entry.max_retries:
                    entry.retries += 1
                    entry.status = "retrying"
                    backoff = 2 ** (entry.retries - 1)
                    logger.info(
                        "Task %s retrying in %ds (attempt %d)",
                        entry.task_id,
                        backoff,
                        entry.retries,
                    )
                    await asyncio.sleep(backoff)
                    entry.status = "queued"
                    await self._queue.put(entry)
                    return
                else:
                    entry.status = "failed"
                    entry.error = str(e)
                    logger.error("Task %s failed: %s", entry.task_id, e)
            finally:
                entry.finished_at = time.time()

    async def submit(
        self,
        func: Callable[..., Coroutine],
        params: dict,
        priority: int = 0,
        timeout: float | None = None,
        max_retries: int = 0,
    ) -> str:
        task_id = str(uuid4())[:8]
        entry = _TaskEntry(
            task_id=task_id,
            func=func,
            params=params,
            priority=priority,
            timeout=timeout or self._default_timeout,
            max_retries=max_retries,
        )
        self._tasks[task_id] = entry
        self._ensure_worker()
        await self._queue.put(entry)
        return task_id

    async def cancel(self, task_id: str) -> bool:
        entry = self._tasks.get(task_id)
        if entry is None:
            return False
        if entry.status in ("completed", "failed", "timeout"):
            return False
        entry._cancelled = True
        entry.status = "cancelled"
        return True

    def get_status(self, task_id: str) -> dict[str, Any]:
        entry = self._tasks.get(task_id)
        if entry is None:
            return {"status": "unknown"}
        return {
            "task_id": entry.task_id,
            "status": entry.status,
            "result": entry.result,
            "error": entry.error,
            "retries": entry.retries,
            "created_at": entry.created_at,
            "started_at": entry.started_at,
            "finished_at": entry.finished_at,
        }
