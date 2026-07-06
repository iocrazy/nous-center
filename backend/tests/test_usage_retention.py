"""usage/task 表保留清理(审查 P1:三表无限增长)。旧行删、新行留。"""
from datetime import datetime, timedelta, timezone

import pytest

from src.models.execution_task import ExecutionTask
from src.models.llm_usage import LLMUsage
from src.models.tts_usage import TTSUsage
from src.services.usage_retention import cleanup_usage


@pytest.mark.asyncio
async def test_cleanup_deletes_old_keeps_recent(db_session):
    old = datetime.now(timezone.utc) - timedelta(days=100)
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    db_session.add_all([
        LLMUsage(model="m", prompt_tokens=1, completion_tokens=1, total_tokens=2, created_at=old),
        LLMUsage(model="m", prompt_tokens=1, completion_tokens=1, total_tokens=2, created_at=recent),
        TTSUsage(engine="cosyvoice2", characters=10, created_at=old),
        TTSUsage(engine="cosyvoice2", characters=10, created_at=recent),
        ExecutionTask(workflow_name="w", status="completed", created_at=old),
        ExecutionTask(workflow_name="w", status="completed", created_at=recent),
    ])
    await db_session.commit()

    deleted = await cleanup_usage(db_session, max_age_days=90)
    assert deleted == {"llm_usage": 1, "tts_usage": 1, "execution_tasks": 1}

    from sqlalchemy import func, select
    for model in (LLMUsage, TTSUsage, ExecutionTask):
        n = await db_session.scalar(select(func.count()).select_from(model))
        assert n == 1, f"{model.__tablename__} 应剩 1 行,实剩 {n}"


@pytest.mark.asyncio
async def test_cleanup_noop_when_all_recent(db_session):
    db_session.add(LLMUsage(model="m", total_tokens=1,
                            created_at=datetime.now(timezone.utc)))
    await db_session.commit()
    deleted = await cleanup_usage(db_session, max_age_days=90)
    assert deleted["llm_usage"] == 0
