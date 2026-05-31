"""RPM/TPM 限流器:准入即占坑(reserve-at-admission)语义回归。

针对 round2 #7 Part 2 的 TOCTOU 修复 —— 旧实现 check(只读)+ record(推理后追加)
之间有窗口,并发请求在 record 前都看到名额未满 → RPM 被绕过;断连/非 LLM 请求不
record → 不计入 RPM。现在 reserve 在同一临界区检查+占坑。
"""

import pytest

from src.errors import RateLimitError
from src.services.rate_limiter import InstanceRateLimiter


@pytest.mark.asyncio
async def test_reserve_occupies_slot_immediately_without_record():
    """reserve 即占坑:连续 reserve 到 rpm 上限后,第 rpm+1 次直接 429,
    全程不调用 record —— 证明 RPM 不再依赖推理后的记账(断连/非 LLM 也算)。"""
    rl = InstanceRateLimiter()
    inst = 1
    for _ in range(3):
        await rl.reserve(inst, rpm_limit=3, tpm_limit=None)
    with pytest.raises(RateLimitError) as ei:
        await rl.reserve(inst, rpm_limit=3, tpm_limit=None)
    assert ei.value.code == "rate_limit_rpm"


@pytest.mark.asyncio
async def test_concurrent_admission_cannot_exceed_rpm():
    """模拟旧 bug:多个请求"同时"准入(record 都还没发生)。
    旧 check 会全放行;现在第 rpm+1 个必被挡。"""
    rl = InstanceRateLimiter()
    inst = 2
    import asyncio

    async def admit():
        try:
            await rl.reserve(inst, rpm_limit=5, tpm_limit=None)
            return True
        except RateLimitError:
            return False

    results = await asyncio.gather(*[admit() for _ in range(20)])
    assert sum(results) == 5  # 恰好 5 个通过,其余 15 被限流


@pytest.mark.asyncio
async def test_tpm_uses_settled_token_window():
    """TPM 看已结算的尾随窗口:record 累计超过 tpm 后,下次 reserve 抛 tpm。"""
    rl = InstanceRateLimiter()
    inst = 3
    # 先合法准入一次并记 token
    await rl.reserve(inst, rpm_limit=None, tpm_limit=1000)
    await rl.record(inst, 1000)
    with pytest.raises(RateLimitError) as ei:
        await rl.reserve(inst, rpm_limit=None, tpm_limit=1000)
    assert ei.value.code == "rate_limit_tpm"


@pytest.mark.asyncio
async def test_no_limits_is_noop():
    """rpm/tpm 都为 None 时 reserve 不占坑、不抛。"""
    rl = InstanceRateLimiter()
    for _ in range(100):
        await rl.reserve(9, rpm_limit=None, tpm_limit=None)
    # 之后设上限仍是干净窗口
    await rl.reserve(9, rpm_limit=1, tpm_limit=None)


@pytest.mark.asyncio
async def test_window_prunes_old_reservations(monkeypatch):
    """超过 60s 的占坑被 prune,名额释放。"""
    import src.services.rate_limiter as rlmod

    rl = InstanceRateLimiter()
    inst = 4
    fake = {"t": 1000.0}
    monkeypatch.setattr(rlmod.time, "monotonic", lambda: fake["t"])
    await rl.reserve(inst, rpm_limit=1, tpm_limit=None)
    with pytest.raises(RateLimitError):
        await rl.reserve(inst, rpm_limit=1, tpm_limit=None)
    # 推进超过窗口
    fake["t"] = 1000.0 + rlmod.WINDOW_SECONDS + 1
    await rl.reserve(inst, rpm_limit=1, tpm_limit=None)  # 不再抛
