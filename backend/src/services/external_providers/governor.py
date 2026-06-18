"""ProviderGovernor — 外部 CLI provider 的并发/节流护栏。

封号护栏的核心:把单账号在云侧看到的请求形态锁在「单人、低频、串行」。四道闸:

1. **并发** asyncio.Semaphore(concurrency):同时在飞的调用数上限(默认 1)。
2. **队列容量** queue_capacity:等待槽位的请求数上限,超了立刻 GovernorBusyError(→ 503),
   不无限堆积(对齐 GroupScheduler 满→503 语义)。
3. **限速** 令牌桶 rate_per_min:每分钟最多发起 N 次(带 burst)。
4. **节流** min_interval_s:两次调用「发起」之间的最小间隔。

另含可选 **结果缓存**(cache_ttl_s>0):相同 cache_key 命中直接返上次结果,省一次云调用。

时钟与 sleep 可注入(clock / sleep),让限速/节流行为在单测里确定性可验,不依赖真实时间。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


class GovernorBusyError(RuntimeError):
    """等待队列已满,拒绝新请求(路由层映射 503)。"""

    def __init__(self, provider: str, capacity: int) -> None:
        super().__init__(f"{provider} 排队已满(容量 {capacity}),请稍后再试")
        self.provider = provider
        self.capacity = capacity


class _TokenBucket:
    """令牌桶:capacity 个 burst,按 rate_per_sec 持续回填。"""

    def __init__(self, rate_per_min: float, clock: Callable[[], float]) -> None:
        self._rate_per_sec = max(0.0, rate_per_min) / 60.0
        self._capacity = max(1.0, float(rate_per_min)) if rate_per_min > 0 else 0.0
        self._tokens = self._capacity
        self._clock = clock
        self._last = clock()

    @property
    def enabled(self) -> bool:
        return self._rate_per_sec > 0

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)

    def time_until_token(self) -> float:
        """还要等多少秒才有 1 个令牌(0 = 现在就有)。"""
        if not self.enabled:
            return 0.0
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / self._rate_per_sec

    def consume(self) -> None:
        if self.enabled:
            self._tokens -= 1.0


class ProviderGovernor:
    def __init__(
        self,
        provider: str,
        *,
        concurrency: int = 1,
        rate_per_min: float = 0.0,
        min_interval_s: float = 0.0,
        queue_capacity: int = 16,
        cache_ttl_s: float = 0.0,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.provider = provider
        self.concurrency = max(1, int(concurrency))
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.queue_capacity = max(0, int(queue_capacity))
        self.cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._sem = asyncio.Semaphore(self.concurrency)
        self._bucket = _TokenBucket(rate_per_min, self._clock)
        self._last_start = 0.0
        self._started_once = False
        self._waiting = 0
        self._cache: dict[str, tuple[float, Any]] = {}

    # ---- 缓存 ------------------------------------------------------------

    def _cache_get(self, key: str | None) -> Any | None:
        if not key or self.cache_ttl_s <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires, value = entry
        if self._clock() >= expires:
            self._cache.pop(key, None)
            return None
        return value

    def _cache_put(self, key: str | None, value: Any) -> None:
        if not key or self.cache_ttl_s <= 0:
            return
        self._cache[key] = (self._clock() + self.cache_ttl_s, value)

    # ---- 主入口 ----------------------------------------------------------

    async def run(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        cache_key: str | None = None,
    ) -> Any:
        """经四道闸 + 缓存执行 factory()。

        factory 是无参 async 工厂(每次重试/调用都新建 coroutine)。
        """
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 队列容量:已在等待的请求数 >= capacity 时直接拒。waiting 计的是「卡在闸上」的请求,
        # 不含正在执行的(那些占的是 concurrency 槽)。
        if self._waiting >= self.queue_capacity and self._sem.locked():
            raise GovernorBusyError(self.provider, self.queue_capacity)

        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1

        try:
            await self._gate_throttle_and_rate()
            self._last_start = self._clock()
            self._started_once = True
            self._bucket.consume()
            result = await factory()
            self._cache_put(cache_key, result)
            return result
        finally:
            self._sem.release()

    async def _gate_throttle_and_rate(self) -> None:
        """阻塞直到「最小间隔」和「令牌桶」两个条件都满足。"""
        # 节流:距上次发起不足 min_interval_s 则补足。
        if self.min_interval_s > 0 and self._started_once:
            wait = self.min_interval_s - (self._clock() - self._last_start)
            if wait > 0:
                await self._sleep(wait)
        # 限速:令牌桶没令牌则等到有。
        wait_token = self._bucket.time_until_token()
        if wait_token > 0:
            await self._sleep(wait_token)
