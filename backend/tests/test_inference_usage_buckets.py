"""get_inference_usage 的时间分桶(date_trunc)在真 PG 上按 day/hour 正确落桶。

前身是 round4 #4 的「SQLite 上不再 500」用例;全局改 PostgreSQL 后 strftime 分支已删,
本用例改为验证 PG 路径的分桶结果。"""

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models.llm_usage import LLMUsage


@pytest.mark.asyncio
async def test_get_inference_usage_buckets_by_day_and_hour(pg_engine, monkeypatch):
    from src.services import usage_service

    engine = pg_engine   # 表由 fixture 建好
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as s:
        s.add(LLMUsage(
            model="m", instance_id=1, prompt_tokens=10, completion_tokens=5,
            total_tokens=15, created_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        ))
        await s.commit()

    monkeypatch.setattr(usage_service, "get_session_factory", lambda: sf)

    # 显式 start/end 框住 2026-05-30 的数据 —— 否则默认窗口是「now-7天」,随系统日期推移
    # 这条固定日期的数据会滑出窗口(2026-06-06 跑时 now-7d 刚好卡在 05-30 → DataCount 0)。
    win_start = datetime(2026, 5, 29, tzinfo=timezone.utc)
    win_end = datetime(2026, 5, 31, tzinfo=timezone.utc)

    res = await usage_service.get_inference_usage(
        interval="day", columnar=True, start=win_start, end=win_end)
    assert res["DataCount"] == 1
    row = res["Data"][0]
    # date_trunc('day') 返 datetime → _iso_bucket 给 "2026-05-30T00:00:00+00:00"
    assert row[0].startswith("2026-05-30")
    assert int(row[2]) == 10 and int(row[3]) == 5

    # 非 columnar 也不崩
    res2 = await usage_service.get_inference_usage(interval="hour", start=win_start, end=win_end)
    assert res2["data"][0]["hour"].startswith("2026-05-30T12:00")

