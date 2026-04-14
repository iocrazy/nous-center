"""Unified usage recording and querying for LLM + TTS."""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.llm_usage import LLMUsage
from src.models.tts_usage import TTSUsage
from src.models.database import create_session_factory

logger = logging.getLogger(__name__)


async def record_llm_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: int | None = None,
    instance_id: int | None = None,
    api_key_id: int | None = None,
) -> None:
    """Record an LLM inference event."""
    sf = create_session_factory()
    async with sf() as session:
        usage = LLMUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration_ms=duration_ms,
            instance_id=instance_id,
            api_key_id=api_key_id,
        )
        session.add(usage)
        await session.commit()


async def record_tts_usage(
    engine: str,
    characters: int,
    duration_ms: int | None = None,
    rtf: float | None = None,
    cached: bool = False,
) -> None:
    """Record a TTS synthesis event."""
    sf = create_session_factory()
    async with sf() as session:
        usage = TTSUsage(
            engine=engine,
            characters=characters,
            duration_ms=duration_ms,
            rtf=rtf,
            cached=cached,
        )
        session.add(usage)
        await session.commit()


async def get_usage_summary(since: datetime | None = None) -> dict:
    """Get aggregated usage stats."""
    if since is None:
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    sf = create_session_factory()
    async with sf() as session:
        # LLM usage
        llm_result = await session.execute(
            select(
                func.count().label("calls"),
                func.coalesce(func.sum(LLMUsage.prompt_tokens), 0).label("prompt_tokens"),
                func.coalesce(func.sum(LLMUsage.completion_tokens), 0).label("completion_tokens"),
                func.coalesce(func.sum(LLMUsage.total_tokens), 0).label("total_tokens"),
            ).where(LLMUsage.created_at >= since)
        )
        llm = llm_result.one()

        # TTS usage
        tts_result = await session.execute(
            select(
                func.count().label("calls"),
                func.coalesce(func.sum(TTSUsage.characters), 0).label("characters"),
            ).where(TTSUsage.created_at >= since)
        )
        tts = tts_result.one()

        # All-time totals
        llm_total = await session.execute(
            select(
                func.count().label("calls"),
                func.coalesce(func.sum(LLMUsage.total_tokens), 0).label("total_tokens"),
            )
        )
        llm_all = llm_total.one()

        return {
            "today": {
                "llm_calls": llm.calls,
                "llm_prompt_tokens": llm.prompt_tokens,
                "llm_completion_tokens": llm.completion_tokens,
                "llm_total_tokens": llm.total_tokens,
                "tts_calls": tts.calls,
                "tts_characters": tts.characters,
                "total_calls": llm.calls + tts.calls,
            },
            "all_time": {
                "llm_calls": llm_all.calls,
                "llm_total_tokens": llm_all.total_tokens,
            },
        }


async def get_usage_by_model(since: datetime | None = None) -> list[dict]:
    """Get per-model usage breakdown."""
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=7)

    sf = create_session_factory()
    async with sf() as session:
        result = await session.execute(
            select(
                LLMUsage.model,
                func.count().label("calls"),
                func.sum(LLMUsage.prompt_tokens).label("prompt_tokens"),
                func.sum(LLMUsage.completion_tokens).label("completion_tokens"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
            )
            .where(LLMUsage.created_at >= since)
            .group_by(LLMUsage.model)
            .order_by(func.sum(LLMUsage.total_tokens).desc())
        )
        return [
            {
                "model": row.model,
                "calls": row.calls,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
            }
            for row in result
        ]
