"""Unified usage recording and querying for LLM + TTS."""

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select

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
    if instance_id is not None:
        from src.services.rate_limiter import get_rate_limiter
        await get_rate_limiter().record(
            instance_id, prompt_tokens + completion_tokens
        )


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


async def get_inference_usage(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = "day",
    group_by: str = "Model",
    instance_id: int | None = None,
    model: str | None = None,
    columnar: bool = False,
) -> dict:
    """Ark-style inference usage query.

    interval: "day" | "hour" — bucket width.
    group_by: "Model" | "Instance" | "ApiKey".
    Dialect note: uses `date_trunc` (PG). SQLite tests should mock.
    """
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=7)
    if interval not in ("day", "hour"):
        raise ValueError("interval must be 'day' or 'hour'")
    group_by = group_by.lower()
    group_col_map = {
        "model": LLMUsage.model,
        "instance": LLMUsage.instance_id,
        "apikey": LLMUsage.api_key_id,
    }
    if group_by not in group_col_map:
        raise ValueError("group_by must be Model|Instance|ApiKey")
    gcol = group_col_map[group_by]

    bucket = func.date_trunc(interval, LLMUsage.created_at).label("bucket")

    sf = create_session_factory()
    async with sf() as session:
        stmt = (
            select(
                bucket,
                gcol.label("group_key"),
                func.count().label("req_cnt"),
                func.coalesce(func.sum(LLMUsage.prompt_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(LLMUsage.completion_tokens), 0).label("output_tokens"),
            )
            .where(LLMUsage.created_at >= start, LLMUsage.created_at < end)
            .group_by(bucket, gcol)
            .order_by(bucket.asc())
        )
        if instance_id is not None:
            stmt = stmt.where(LLMUsage.instance_id == instance_id)
        if model is not None:
            stmt = stmt.where(LLMUsage.model == model)

        rows = (await session.execute(stmt)).all()

    time_field = "Day" if interval == "day" else "Hour"
    group_field = {"model": "Model", "instance": "Instance", "apikey": "ApiKey"}[group_by]

    if columnar:
        return {
            "Fields": [
                {"Name": time_field, "Type": "DATE" if interval == "day" else "DATETIME"},
                {"Name": group_field, "Type": "STRING"},
                {"Name": "InputTokens", "Type": "BIGINT"},
                {"Name": "OutputTokens", "Type": "BIGINT"},
                {"Name": "ReqCnt", "Type": "BIGINT"},
            ],
            "Data": [
                [
                    r.bucket.isoformat() if r.bucket else None,
                    str(r.group_key) if r.group_key is not None else None,
                    int(r.input_tokens),
                    int(r.output_tokens),
                    int(r.req_cnt),
                ]
                for r in rows
            ],
            "DataCount": len(rows),
        }
    return {
        "interval": interval,
        "group_by": group_field,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "data": [
            {
                time_field.lower(): r.bucket.isoformat() if r.bucket else None,
                group_field.lower(): r.group_key,
                "input_tokens": int(r.input_tokens),
                "output_tokens": int(r.output_tokens),
                "req_cnt": int(r.req_cnt),
            }
            for r in rows
        ],
    }
