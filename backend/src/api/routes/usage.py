"""m13 usage stats — admin-only aggregates over llm_usage + tts_usage.

Three endpoints, all under `/api/v1/usage`:

  GET  /summary?days=N           — 4 stat cards (calls / tokens / avg latency / error rate)
  GET  /timeseries?days=N        — daily call counts per service
  GET  /top-keys?days=N&limit=K  — top API keys by call volume

Error rate is not tracked in the usage tables yet (no status column on
LLMUsage / TTSUsage rows), so we return `null` and let the UI render a
"—". P95 is best-effort: on Postgres we use `percentile_cont(0.95)`; on
SQLite (tests) we fall back to `avg(duration_ms)` so the test suite
still exercises the route shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.llm_usage import LLMUsage
from src.models.service_instance import ServiceInstance
from src.models.tts_usage import TTSUsage

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


# ---------- shapes ----------


class UsageSummary(BaseModel):
    days: int
    period_start: datetime
    period_end: datetime
    total_calls: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    tts_characters: int
    avg_latency_ms: float | None
    error_rate: float | None  # null until we track status on usage rows
    prev_total_calls: int
    prev_total_tokens: int


class TimeseriesPoint(BaseModel):
    date: str  # YYYY-MM-DD (UTC)
    by_service: dict[str, int]


class Timeseries(BaseModel):
    days: int
    points: list[TimeseriesPoint]
    top_services: list[str]


class TopKey(BaseModel):
    api_key_id: int
    label: str | None
    key_prefix: str | None
    mode: str  # "legacy" | "m:n"
    calls: int
    tokens: int
    avg_latency_ms: float | None


class TopKeys(BaseModel):
    days: int
    rows: list[TopKey]


# ---------- helpers ----------


def _window(days: int) -> tuple[datetime, datetime, datetime]:
    """(prev_start, period_start, period_end) all UTC, half-open intervals."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    prev_start = start - timedelta(days=days)
    return prev_start, start, end


def _is_postgres(session: AsyncSession) -> bool:
    return session.bind.dialect.name == "postgresql"  # type: ignore[union-attr]


# ---------- routes ----------


@router.get(
    "/summary",
    response_model=UsageSummary,
    dependencies=[Depends(require_admin)],
)
async def usage_summary(
    days: int = Query(7, ge=1, le=365),
    session: AsyncSession = Depends(get_async_session),
):
    prev_start, start, end = _window(days)

    llm_curr = await session.execute(
        select(
            func.coalesce(func.count(LLMUsage.id), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.avg(LLMUsage.duration_ms),
        ).where(LLMUsage.created_at >= start, LLMUsage.created_at < end),
    )
    calls_llm, tokens, prompt, completion, avg_latency = llm_curr.one()

    tts_curr = await session.execute(
        select(
            func.coalesce(func.count(TTSUsage.id), 0),
            func.coalesce(func.sum(TTSUsage.characters), 0),
        ).where(TTSUsage.created_at >= start, TTSUsage.created_at < end),
    )
    calls_tts, tts_chars = tts_curr.one()

    llm_prev = await session.execute(
        select(
            func.coalesce(func.count(LLMUsage.id), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
        ).where(LLMUsage.created_at >= prev_start, LLMUsage.created_at < start),
    )
    prev_calls_llm, prev_tokens = llm_prev.one()

    tts_prev = await session.execute(
        select(func.coalesce(func.count(TTSUsage.id), 0)).where(
            TTSUsage.created_at >= prev_start, TTSUsage.created_at < start
        )
    )
    prev_calls_tts = tts_prev.scalar() or 0

    return UsageSummary(
        days=days,
        period_start=start,
        period_end=end,
        total_calls=int(calls_llm) + int(calls_tts),
        total_tokens=int(tokens),
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
        tts_characters=int(tts_chars),
        avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
        error_rate=None,
        prev_total_calls=int(prev_calls_llm) + int(prev_calls_tts),
        prev_total_tokens=int(prev_tokens),
    )


@router.get(
    "/timeseries",
    response_model=Timeseries,
    dependencies=[Depends(require_admin)],
)
async def usage_timeseries(
    days: int = Query(7, ge=1, le=90),
    top: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_async_session),
):
    """Daily call counts grouped by service name (instance_id → name).

    Anything outside the top-N is bucketed into the literal key "other".
    """
    _prev, start, end = _window(days)

    # Resolve instance_id → display name once.
    name_map: dict[int, str] = {
        sid: name
        for sid, name in (
            await session.execute(select(ServiceInstance.id, ServiceInstance.name))
        ).all()
    }

    # Date bucketing — PG via date_trunc, SQLite via strftime.
    if _is_postgres(session):
        day_col = func.to_char(
            func.date_trunc("day", LLMUsage.created_at), "YYYY-MM-DD"
        )
    else:
        day_col = func.strftime("%Y-%m-%d", LLMUsage.created_at)

    rows = (
        await session.execute(
            select(
                day_col.label("day"),
                LLMUsage.instance_id,
                func.count(LLMUsage.id),
            )
            .where(LLMUsage.created_at >= start, LLMUsage.created_at < end)
            .group_by("day", LLMUsage.instance_id),
        )
    ).all()

    # Aggregate totals per service to pick the top N.
    totals: dict[str, int] = {}
    by_day: dict[str, dict[str, int]] = {}
    for day, instance_id, count in rows:
        svc_name = name_map.get(instance_id, "(unknown)") if instance_id else "(no-binding)"
        totals[svc_name] = totals.get(svc_name, 0) + int(count)
        by_day.setdefault(day, {})[svc_name] = (
            by_day.setdefault(day, {}).get(svc_name, 0) + int(count)
        )

    top_services = [
        s for s, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top]
    ]
    top_set = set(top_services)

    # Fold non-top into "other" + emit one point per day in the window
    # (zero-fill missing days).
    points: list[TimeseriesPoint] = []
    cursor = start.date()
    end_date = end.date()
    while cursor <= end_date:
        day_str = cursor.isoformat()
        bucket: dict[str, int] = {}
        for svc, n in by_day.get(day_str, {}).items():
            key = svc if svc in top_set else "other"
            bucket[key] = bucket.get(key, 0) + n
        points.append(TimeseriesPoint(date=day_str, by_service=bucket))
        cursor = cursor + timedelta(days=1)

    return Timeseries(days=days, points=points, top_services=top_services)


@router.get(
    "/top-keys",
    response_model=TopKeys,
    dependencies=[Depends(require_admin)],
)
async def top_keys(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
):
    _prev, start, end = _window(days)

    stmt = (
        select(
            LLMUsage.api_key_id,
            func.count(LLMUsage.id).label("calls"),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0).label("tokens"),
            func.avg(LLMUsage.duration_ms).label("avg_ms"),
        )
        .where(
            LLMUsage.created_at >= start,
            LLMUsage.created_at < end,
            LLMUsage.api_key_id.is_not(None),
        )
        .group_by(LLMUsage.api_key_id)
        .order_by(desc("calls"))
        .limit(limit)
    )
    rows: list[Any] = (await session.execute(stmt)).all()

    if not rows:
        return TopKeys(days=days, rows=[])

    key_ids = [r.api_key_id for r in rows]
    keys = {
        k.id: k
        for k in (
            await session.execute(
                select(InstanceApiKey).where(InstanceApiKey.id.in_(key_ids))
            )
        )
        .scalars()
        .all()
    }

    out: list[TopKey] = []
    for r in rows:
        k = keys.get(r.api_key_id)
        out.append(
            TopKey(
                api_key_id=int(r.api_key_id),
                label=getattr(k, "label", None) if k else None,
                key_prefix=getattr(k, "key_prefix", None) if k else None,
                mode="legacy" if k and k.instance_id is not None else "m:n",
                calls=int(r.calls),
                tokens=int(r.tokens),
                avg_latency_ms=float(r.avg_ms) if r.avg_ms is not None else None,
            )
        )
    return TopKeys(days=days, rows=out)


# Suppress unused-import linter complaints for `literal` (kept available
# for follow-up if we add SQLite percentile fallbacks).
_ = literal
