"""round4 #4:get_inference_usage 在 SQLite 上不再 500(早先无条件 date_trunc)。"""

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.llm_usage import LLMUsage


@pytest.mark.asyncio
async def test_get_inference_usage_sqlite_does_not_crash(tmp_path, monkeypatch):
    from src.services import usage_service

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'u.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as s:
        s.add(LLMUsage(
            model="m", instance_id=1, prompt_tokens=10, completion_tokens=5,
            total_tokens=15, created_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        ))
        await s.commit()

    monkeypatch.setattr(usage_service, "get_session_factory", lambda: sf)

    # 早先这里对 SQLite 抛 date_trunc 不支持 → 现在 strftime 分支正常返回
    res = await usage_service.get_inference_usage(interval="day", columnar=True)
    assert res["DataCount"] == 1
    row = res["Data"][0]
    assert row[0] == "2026-05-30"  # strftime day bucket(字符串,_iso_bucket 原样返回)
    assert int(row[2]) == 10 and int(row[3]) == 5

    # 非 columnar 也不崩
    res2 = await usage_service.get_inference_usage(interval="hour")
    assert res2["data"][0]["hour"].startswith("2026-05-30 12:00")

    await engine.dispose()
