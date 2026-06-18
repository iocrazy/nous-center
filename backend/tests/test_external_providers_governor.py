"""ProviderGovernor 护栏行为(注入时钟/sleep,确定性验证)。"""
from __future__ import annotations

import asyncio

import pytest

from src.services.external_providers.governor import (
    GovernorBusyError,
    ProviderGovernor,
    _TokenBucket,
)


class FakeClock:
    """可注入时钟:sleep 直接快进虚拟时间,记录每次 sleep 时长。"""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.t += secs


def _gov(**kw):
    clock = FakeClock()
    gov = ProviderGovernor("test", clock=clock.now, sleep=clock.sleep, **kw)
    return gov, clock


async def test_min_interval_throttle_inserts_wait():
    gov, clock = _gov(concurrency=1, min_interval_s=5.0)

    async def factory():
        return "ok"

    await gov.run(factory)          # 首次:不节流
    assert clock.slept == []
    await gov.run(factory)          # 第二次:距上次 0s < 5s,应补睡 5s
    assert clock.slept == [pytest.approx(5.0)]


async def test_concurrency_serializes():
    gov, _clock = _gov(concurrency=1)
    started = []
    release = asyncio.Event()

    async def factory(tag):
        started.append(tag)
        await release.wait()
        return tag

    t1 = asyncio.create_task(gov.run(lambda: factory("a")))
    await asyncio.sleep(0)           # 让 t1 进入 factory
    t2 = asyncio.create_task(gov.run(lambda: factory("b")))
    await asyncio.sleep(0)
    assert started == ["a"]          # concurrency=1:b 还没启动
    release.set()
    await asyncio.gather(t1, t2)
    assert started == ["a", "b"]


async def test_queue_capacity_rejects_when_full():
    gov, _clock = _gov(concurrency=1, queue_capacity=0)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def long_factory():
        entered.set()
        await release.wait()
        return "done"

    t1 = asyncio.create_task(gov.run(long_factory))
    await entered.wait()             # 确保 t1 占住唯一并发槽
    with pytest.raises(GovernorBusyError):
        await gov.run(long_factory)  # 容量 0 + 槽被占 → 立刻拒
    release.set()
    assert await t1 == "done"


async def test_cache_hit_skips_factory():
    gov, _clock = _gov(concurrency=1, cache_ttl_s=60.0)
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        return calls

    r1 = await gov.run(factory, cache_key="k")
    r2 = await gov.run(factory, cache_key="k")
    assert r1 == 1 and r2 == 1       # 第二次命中缓存,factory 未再调
    assert calls == 1


async def test_cache_expires():
    clock = FakeClock()
    gov = ProviderGovernor("t", cache_ttl_s=10.0, clock=clock.now, sleep=clock.sleep)
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        return calls

    await gov.run(factory, cache_key="k")
    clock.t += 11.0                  # 过期
    await gov.run(factory, cache_key="k")
    assert calls == 2


def test_token_bucket_refills_over_time():
    clock = FakeClock()
    bucket = _TokenBucket(rate_per_min=6.0, clock=clock.now)  # capacity 6, 0.1/s
    for _ in range(6):
        assert bucket.time_until_token() == 0.0
        bucket.consume()
    wait = bucket.time_until_token()  # 桶空,需等 1/0.1 = 10s
    assert wait == pytest.approx(10.0)
    clock.t += 10.0
    assert bucket.time_until_token() == 0.0


def test_token_bucket_disabled_when_rate_zero():
    clock = FakeClock()
    bucket = _TokenBucket(rate_per_min=0.0, clock=clock.now)
    assert not bucket.enabled
    for _ in range(100):
        assert bucket.time_until_token() == 0.0
        bucket.consume()
