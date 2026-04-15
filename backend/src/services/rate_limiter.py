"""In-process sliding-window rate limiter for service instances.

Single-worker scope only. Multi-worker / multi-host needs Redis (v2).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from src.errors import RateLimitError

WINDOW_SECONDS = 60


class InstanceRateLimiter:
    def __init__(self):
        # instance_id -> deque[(ts, tokens)]
        self._events: dict[int, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    def _prune(self, q: deque, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while q and q[0][0] < cutoff:
            q.popleft()

    async def check(
        self,
        instance_id: int,
        rpm_limit: int | None,
        tpm_limit: int | None,
    ) -> None:
        if not rpm_limit and not tpm_limit:
            return
        now = time.monotonic()
        async with self._lock:
            q = self._events[instance_id]
            self._prune(q, now)
            if rpm_limit and len(q) >= rpm_limit:
                raise RateLimitError(
                    f"instance RPM limit ({rpm_limit}/min) exceeded",
                    code="rate_limit_rpm",
                )
            if tpm_limit:
                used = sum(tok for _, tok in q)
                if used >= tpm_limit:
                    raise RateLimitError(
                        f"instance TPM limit ({tpm_limit}/min) exceeded; "
                        f"current window: {used}",
                        code="rate_limit_tpm",
                    )

    async def record(self, instance_id: int, tokens: int) -> None:
        now = time.monotonic()
        async with self._lock:
            q = self._events[instance_id]
            self._prune(q, now)
            q.append((now, max(0, int(tokens or 0))))


_limiter: InstanceRateLimiter | None = None


def get_rate_limiter() -> InstanceRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = InstanceRateLimiter()
    return _limiter
