"""m04 dashboard summary — one-shot business overview.

Pulls from llm_usage / tts_usage / service_instances / instance_api_keys /
api_key_grants / alert_rules / resource_packs in a single round trip so
the m04 page doesn't need 6 React Query hooks. Admin-only.

Returned shape is intentionally flat: each card on the page reads one
field. Anything that's not collected yet (e.g. error_rate) returns null
and the UI shows a placeholder.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.llm_usage import LLMUsage
from src.models.service_instance import ServiceInstance
from src.models.tts_usage import TTSUsage

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


class TopServiceRow(BaseModel):
    service_name: str
    calls: int
    percent: float  # of today's total LLM calls; 0..100


class AlertItem(BaseModel):
    id: int
    grant_id: int
    service_name: str | None
    threshold_percent: int
    last_notified_at: datetime | None
    severity: str  # "warn" | "err"


class DashboardSummary(BaseModel):
    today_calls: int
    today_calls_delta_pct: float | None  # vs yesterday
    month_tokens: int
    month_tokens_quota: int | None  # null if no packs
    month_tokens_used_pct: float | None
    active_alerts_count: int
    active_alerts_top_label: str | None
    api_key_count: int
    service_count: int
    unbound_key_count: int
    top_services_today: list[TopServiceRow]
    recent_alerts: list[AlertItem]


@router.get(
    "/summary",
    response_model=DashboardSummary,
    dependencies=[Depends(require_admin)],
)
async def dashboard_summary(
    top: int = Query(5, ge=1, le=20),
    alerts: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_async_session),
):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yest_start = today_start - timedelta(days=1)
    month_start = today_start - timedelta(days=30)

    today_llm, today_tts = await _calls_in(session, today_start, now)
    yest_llm, yest_tts = await _calls_in(session, yest_start, today_start)
    today_total = today_llm + today_tts
    yest_total = yest_llm + yest_tts
    delta_pct = (
        ((today_total - yest_total) / yest_total) * 100.0 if yest_total > 0 else None
    )

    month_tokens = (
        await session.execute(
            select(func.coalesce(func.sum(LLMUsage.total_tokens), 0)).where(
                LLMUsage.created_at >= month_start, LLMUsage.created_at < now
            )
        )
    ).scalar() or 0

    # Token quota: sum total_units of resource_packs whose grants are on
    # a service with meter_dim='tokens'. None of this is enforced today —
    # we just surface the headroom number.
    quota_row = (
        await session.execute(
            select(
                func.coalesce(func.sum(ResourcePack.total_units), 0),
            )
            .select_from(ResourcePack)
            .join(ApiKeyGrant, ApiKeyGrant.id == ResourcePack.grant_id)
            .join(ServiceInstance, ServiceInstance.id == ApiKeyGrant.service_id)
            .where(ServiceInstance.meter_dim == "tokens")
        )
    ).scalar()
    month_tokens_quota = int(quota_row) if quota_row else None
    month_tokens_used_pct = (
        round((int(month_tokens) / month_tokens_quota) * 100.0, 1)
        if month_tokens_quota and month_tokens_quota > 0
        else None
    )

    # Top services today: group LLM calls by instance_id → service name.
    name_map: dict[int, str] = {
        sid: name
        for sid, name in (
            await session.execute(select(ServiceInstance.id, ServiceInstance.name))
        ).all()
    }
    by_svc_today = (
        await session.execute(
            select(
                LLMUsage.instance_id,
                func.count(LLMUsage.id).label("c"),
            )
            .where(
                LLMUsage.created_at >= today_start,
                LLMUsage.created_at < now,
                LLMUsage.instance_id.is_not(None),
            )
            .group_by(LLMUsage.instance_id)
            .order_by(desc("c"))
            .limit(top)
        )
    ).all()
    today_total_for_pct = today_llm or 1  # avoid div-by-zero
    top_services_today = [
        TopServiceRow(
            service_name=name_map.get(row.instance_id, "(unknown)"),
            calls=int(row.c),
            percent=round((int(row.c) / today_total_for_pct) * 100.0, 1),
        )
        for row in by_svc_today
    ]

    api_key_count = (
        await session.execute(select(func.count(InstanceApiKey.id)))
    ).scalar() or 0
    service_count = (
        await session.execute(select(func.count(ServiceInstance.id)))
    ).scalar() or 0
    # "Unbound" v3-style: keys with no active grant and (legacy) no
    # bound instance_id. Counts the keys that exist but reach nothing.
    unbound_key_count = (
        await session.execute(
            select(func.count(InstanceApiKey.id)).where(
                InstanceApiKey.instance_id.is_(None),
                ~InstanceApiKey.id.in_(
                    select(ApiKeyGrant.api_key_id).where(
                        ApiKeyGrant.status == "active"
                    )
                ),
            )
        )
    ).scalar() or 0

    # Active alerts: alert_rules with last_notified_at within the past 7
    # days. Severity heuristic: ≥80% threshold == err, else warn.
    alert_window = now - timedelta(days=7)
    alert_rows = (
        await session.execute(
            select(AlertRule, ApiKeyGrant)
            .join(ApiKeyGrant, ApiKeyGrant.id == AlertRule.grant_id)
            .where(
                AlertRule.enabled.is_(True),
                AlertRule.last_notified_at.is_not(None),
                AlertRule.last_notified_at >= alert_window,
            )
            .order_by(desc(AlertRule.last_notified_at))
            .limit(alerts)
        )
    ).all()
    recent_alerts = [
        AlertItem(
            id=int(rule.id),
            grant_id=int(rule.grant_id),
            service_name=name_map.get(grant.service_id),
            threshold_percent=int(rule.threshold_percent),
            last_notified_at=rule.last_notified_at,
            severity="err" if rule.threshold_percent >= 80 else "warn",
        )
        for rule, grant in alert_rows
    ]

    return DashboardSummary(
        today_calls=today_total,
        today_calls_delta_pct=delta_pct,
        month_tokens=int(month_tokens),
        month_tokens_quota=month_tokens_quota,
        month_tokens_used_pct=month_tokens_used_pct,
        active_alerts_count=len(recent_alerts),
        active_alerts_top_label=(
            recent_alerts[0].service_name if recent_alerts else None
        ),
        api_key_count=int(api_key_count),
        service_count=int(service_count),
        unbound_key_count=int(unbound_key_count),
        top_services_today=top_services_today,
        recent_alerts=recent_alerts,
    )


async def _calls_in(
    session: AsyncSession, start: datetime, end: datetime
) -> tuple[int, int]:
    llm = (
        await session.execute(
            select(func.count(LLMUsage.id)).where(
                LLMUsage.created_at >= start, LLMUsage.created_at < end
            )
        )
    ).scalar() or 0
    tts = (
        await session.execute(
            select(func.count(TTSUsage.id)).where(
                TTSUsage.created_at >= start, TTSUsage.created_at < end
            )
        )
    ).scalar() or 0
    return int(llm), int(tts)
