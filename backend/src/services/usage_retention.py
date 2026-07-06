"""usage / task 历史表保留清理 —— 防无限增长(审查 P1)。

llm_usage / tts_usage / execution_tasks 每次推理/任务都写一行,历史上**无任何
retention**(只有 logs / status_samples 有),越用越大、聚合查询越慢。按 created_at
删旧行(三表 created_at 均有索引;execution_tasks 的索引由本次改动补上)。

保留期默认 90 天,可 env NOUS_USAGE_RETENTION_DAYS 覆盖。execution_tasks 删旧任务
= 旧任务引用的图也随之可被 image reaper 回收(图寿命=任务寿命,与既有语义一致)。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.execution_task import ExecutionTask
from src.models.llm_usage import LLMUsage
from src.models.tts_usage import TTSUsage

logger = logging.getLogger(__name__)

_TABLES = (("llm_usage", LLMUsage), ("tts_usage", TTSUsage), ("execution_tasks", ExecutionTask))


def _retention_days() -> int:
    try:
        return int(os.getenv("NOUS_USAGE_RETENTION_DAYS", ""))
    except (TypeError, ValueError):
        return 90


async def cleanup_usage(session: AsyncSession, max_age_days: int | None = None) -> dict[str, int]:
    """删除三张历史表里 created_at 早于 cutoff 的行。返回 {表: 删除行数}。"""
    days = max_age_days if max_age_days is not None else _retention_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted: dict[str, int] = {}
    for name, model in _TABLES:
        res = await session.execute(delete(model).where(model.created_at < cutoff))
        deleted[name] = res.rowcount or 0
    await session.commit()
    total = sum(deleted.values())
    if total:
        logger.info("usage retention: 删除 %d 行(>%d 天): %s", total, days, deleted)
    return deleted
