"""Admin CRUD for ApiKeyGrant + ResourcePack + AlertRule.

The frontend catalog/management pages (/services, /api-management) need
endpoints to read and mutate grants and their attached packs/alerts.
The call path is always:

  admin token (management console)
    → POST /api/v1/keys/{key_id}/grants       (create grant)
    → POST /api/v1/grants/{grant_id}/packs    (add quota pack)
    → POST /api/v1/grants/{grant_id}/alerts   (configure alert rule)

Read endpoints are admin-gated too: these shapes expose used/total and
alert state that's not meant for the bearer-token surface. A separate
`GET /api/v1/services/me` serves the bearer-token client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_serializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.api.deps_auth import verify_bearer_token_any
from src.models.api_gateway import AlertRule, ApiKeyGrant, ResourcePack
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

router = APIRouter(prefix="/api/v1", tags=["api-gateway"])


# ---------- Pydantic shapes ----------


class GrantCreate(BaseModel):
    instance_id: int


class GrantOut(BaseModel):
    id: int
    api_key_id: int
    instance_id: int
    instance_name: str
    status: str
    activated_at: datetime
    paused_at: datetime | None = None
    retired_at: datetime | None = None

    # Snowflake IDs → string in JSON to keep round-trip safe in JS.
    @field_serializer("id", "api_key_id", "instance_id", when_used="json")
    def _id_to_str(self, v: int) -> str:
        return str(v)


class GrantPatch(BaseModel):
    status: Literal["active", "paused", "retired"]


class PackCreate(BaseModel):
    name: str
    total_units: int = Field(..., gt=0)
    expires_at: datetime | None = None
    source: Literal["purchased", "free_trial"] = "purchased"


class PackOut(BaseModel):
    id: int
    grant_id: int
    name: str
    total_units: int
    used_units: int
    remaining_units: int
    expires_at: datetime | None = None
    purchased_at: datetime
    source: str


class AlertCreate(BaseModel):
    threshold_percent: int = Field(..., ge=1, le=100)
    pack_id: int | None = None
    enabled: bool = True


class AlertOut(BaseModel):
    id: int
    grant_id: int
    threshold_percent: int
    pack_id: int | None
    enabled: bool
    last_notified_at: datetime | None
    created_at: datetime


class AlertPatch(BaseModel):
    threshold_percent: int | None = Field(None, ge=1, le=100)
    enabled: bool | None = None


# ---------- Admin: grants ----------


@router.post(
    "/keys/{key_id}/grants",
    status_code=201,
    response_model=GrantOut,
    dependencies=[Depends(require_admin)],
)
async def create_grant(
    key_id: int,
    body: GrantCreate,
    session: AsyncSession = Depends(get_async_session),
):
    key = await session.get(InstanceApiKey, key_id)
    if not key:
        raise HTTPException(404, detail="api key not found")
    instance = await session.get(ServiceInstance, body.instance_id)
    if not instance:
        raise HTTPException(404, detail="instance not found")

    # Dedup: one grant per (key, instance).
    existing = await session.scalar(
        select(ApiKeyGrant).where(
            ApiKeyGrant.api_key_id == key_id,
            ApiKeyGrant.service_id == body.instance_id,
        )
    )
    if existing:
        raise HTTPException(409, detail="grant already exists")

    grant = ApiKeyGrant(api_key_id=key_id, service_id=body.instance_id)
    session.add(grant)
    await session.commit()
    await session.refresh(grant)
    return GrantOut(
        id=grant.id, api_key_id=grant.api_key_id, instance_id=grant.service_id,
        instance_name=instance.name, status=grant.status,
        activated_at=grant.activated_at, paused_at=grant.paused_at,
        retired_at=grant.retired_at,
    )


@router.get(
    "/keys/{key_id}/grants",
    response_model=list[GrantOut],
    dependencies=[Depends(require_admin)],
)
async def list_grants(
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        select(ApiKeyGrant, ServiceInstance)
        .join(ServiceInstance, ServiceInstance.id == ApiKeyGrant.service_id)
        .where(ApiKeyGrant.api_key_id == key_id)
        .order_by(ApiKeyGrant.activated_at.desc())
    )
    rows = (await session.execute(stmt)).all()
    return [
        GrantOut(
            id=g.id, api_key_id=g.api_key_id, instance_id=g.service_id,
            instance_name=inst.name, status=g.status,
            activated_at=g.activated_at, paused_at=g.paused_at,
            retired_at=g.retired_at,
        )
        for g, inst in rows
    ]


@router.patch(
    "/grants/{grant_id}",
    response_model=GrantOut,
    dependencies=[Depends(require_admin)],
)
async def update_grant(
    grant_id: int,
    body: GrantPatch,
    session: AsyncSession = Depends(get_async_session),
):
    grant = await session.get(ApiKeyGrant, grant_id)
    if not grant:
        raise HTTPException(404, detail="grant not found")
    instance = await session.get(ServiceInstance, grant.service_id)

    now = datetime.now(timezone.utc)
    grant.status = body.status
    if body.status == "paused":
        grant.paused_at = now
    elif body.status == "retired":
        grant.retired_at = now
    elif body.status == "active":
        grant.paused_at = None
        grant.retired_at = None
    await session.commit()
    await session.refresh(grant)
    return GrantOut(
        id=grant.id, api_key_id=grant.api_key_id, instance_id=grant.service_id,
        instance_name=instance.name if instance else "", status=grant.status,
        activated_at=grant.activated_at, paused_at=grant.paused_at,
        retired_at=grant.retired_at,
    )


@router.delete(
    "/grants/{grant_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_grant(
    grant_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    grant = await session.get(ApiKeyGrant, grant_id)
    if not grant:
        raise HTTPException(404, detail="grant not found")
    await session.delete(grant)
    await session.commit()


# ---------- Admin: packs ----------


def _pack_to_out(p: ResourcePack) -> PackOut:
    return PackOut(
        id=p.id, grant_id=p.grant_id, name=p.name,
        total_units=p.total_units, used_units=p.used_units,
        remaining_units=max(0, p.total_units - p.used_units),
        expires_at=p.expires_at, purchased_at=p.purchased_at,
        source=p.source,
    )


@router.post(
    "/grants/{grant_id}/packs",
    status_code=201,
    response_model=PackOut,
    dependencies=[Depends(require_admin)],
)
async def create_pack(
    grant_id: int,
    body: PackCreate,
    session: AsyncSession = Depends(get_async_session),
):
    grant = await session.get(ApiKeyGrant, grant_id)
    if not grant:
        raise HTTPException(404, detail="grant not found")
    pack = ResourcePack(
        grant_id=grant_id, name=body.name, total_units=body.total_units,
        used_units=0, expires_at=body.expires_at, source=body.source,
    )
    session.add(pack)
    await session.commit()
    await session.refresh(pack)
    return _pack_to_out(pack)


@router.get(
    "/grants/{grant_id}/packs",
    response_model=list[PackOut],
    dependencies=[Depends(require_admin)],
)
async def list_packs(
    grant_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        select(ResourcePack)
        .where(ResourcePack.grant_id == grant_id)
        .order_by(ResourcePack.purchased_at.desc())
    )
    return [_pack_to_out(p) for p in (await session.execute(stmt)).scalars().all()]


@router.delete(
    "/packs/{pack_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_pack(
    pack_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    pack = await session.get(ResourcePack, pack_id)
    if not pack:
        raise HTTPException(404, detail="pack not found")
    await session.delete(pack)
    await session.commit()


# ---------- Admin: alerts ----------


def _alert_to_out(r: AlertRule) -> AlertOut:
    return AlertOut(
        id=r.id, grant_id=r.grant_id,
        threshold_percent=r.threshold_percent, pack_id=r.pack_id,
        enabled=r.enabled, last_notified_at=r.last_notified_at,
        created_at=r.created_at,
    )


@router.post(
    "/grants/{grant_id}/alerts",
    status_code=201,
    response_model=AlertOut,
    dependencies=[Depends(require_admin)],
)
async def create_alert(
    grant_id: int,
    body: AlertCreate,
    session: AsyncSession = Depends(get_async_session),
):
    grant = await session.get(ApiKeyGrant, grant_id)
    if not grant:
        raise HTTPException(404, detail="grant not found")
    rule = AlertRule(
        grant_id=grant_id,
        threshold_percent=body.threshold_percent,
        pack_id=body.pack_id,
        enabled=body.enabled,
    )
    session.add(rule)
    await session.commit()
    await session.refresh(rule)
    return _alert_to_out(rule)


@router.get(
    "/grants/{grant_id}/alerts",
    response_model=list[AlertOut],
    dependencies=[Depends(require_admin)],
)
async def list_alerts(
    grant_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        select(AlertRule)
        .where(AlertRule.grant_id == grant_id)
        .order_by(AlertRule.created_at.desc())
    )
    return [_alert_to_out(r) for r in (await session.execute(stmt)).scalars().all()]


@router.patch(
    "/alerts/{rule_id}",
    response_model=AlertOut,
    dependencies=[Depends(require_admin)],
)
async def update_alert(
    rule_id: int,
    body: AlertPatch,
    session: AsyncSession = Depends(get_async_session),
):
    rule = await session.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(404, detail="alert rule not found")
    if body.threshold_percent is not None:
        rule.threshold_percent = body.threshold_percent
    if body.enabled is not None:
        rule.enabled = body.enabled
    await session.commit()
    await session.refresh(rule)
    return _alert_to_out(rule)


@router.delete(
    "/alerts/{rule_id}",
    status_code=204,
    dependencies=[Depends(require_admin)],
)
async def delete_alert(
    rule_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    rule = await session.get(AlertRule, rule_id)
    if not rule:
        raise HTTPException(404, detail="alert rule not found")
    await session.delete(rule)
    await session.commit()


# ---------- Bearer-token: my services ----------


class MyServiceOut(BaseModel):
    instance_id: int
    instance_name: str
    category: str | None
    meter_dim: str | None
    grant_status: str
    total_units: int
    used_units: int
    remaining_units: int


class CatalogServiceOut(BaseModel):
    instance_id: int
    instance_name: str
    type: str
    category: str | None
    meter_dim: str | None
    status: str
    total_grants: int
    active_grants: int
    total_units: int
    used_units: int
    remaining_units: int


@router.get(
    "/services/catalog",
    response_model=list[CatalogServiceOut],
    dependencies=[Depends(require_admin)],
)
async def services_catalog(
    session: AsyncSession = Depends(get_async_session),
):
    """Admin catalog: every ServiceInstance + aggregate quota across all
    grants. Used by the admin /services page to render a full-system
    overview. Bearer-token users should hit /services/me instead."""
    from sqlalchemy import func

    # Collect all instances first.
    instances = (await session.execute(
        select(ServiceInstance).order_by(ServiceInstance.name)
    )).scalars().all()

    # Per-instance aggregates: grant counts + pack sums.
    # Split into two scalar queries to keep portability between PG and
    # SQLite (boolean-to-int casting works differently across drivers).
    totals_stmt = (
        select(ApiKeyGrant.service_id, func.count(ApiKeyGrant.id))
        .group_by(ApiKeyGrant.service_id)
    )
    totals = {
        inst_id: count
        for inst_id, count in (await session.execute(totals_stmt)).all()
    }
    actives_stmt = (
        select(ApiKeyGrant.service_id, func.count(ApiKeyGrant.id))
        .where(ApiKeyGrant.status == "active")
        .group_by(ApiKeyGrant.service_id)
    )
    actives = {
        inst_id: count
        for inst_id, count in (await session.execute(actives_stmt)).all()
    }
    grant_counts = {
        inst_id: (totals.get(inst_id, 0), actives.get(inst_id, 0))
        for inst_id in set(totals) | set(actives)
    }

    pack_sums_stmt = (
        select(
            ApiKeyGrant.service_id,
            func.coalesce(func.sum(ResourcePack.total_units), 0).label("total_units"),
            func.coalesce(func.sum(ResourcePack.used_units), 0).label("used_units"),
        )
        .select_from(ResourcePack)
        .join(ApiKeyGrant, ApiKeyGrant.id == ResourcePack.grant_id)
        .group_by(ApiKeyGrant.service_id)
    )
    pack_sums = {
        row.service_id: (row.total_units or 0, row.used_units or 0)
        for row in (await session.execute(pack_sums_stmt)).all()
    }

    out = []
    for inst in instances:
        total_grants, active_grants = grant_counts.get(inst.id, (0, 0))
        total_units, used_units = pack_sums.get(inst.id, (0, 0))
        out.append(CatalogServiceOut(
            instance_id=inst.id,
            instance_name=inst.name,
            type=inst.type or "",
            category=inst.category,
            meter_dim=inst.meter_dim,
            status=inst.status,
            total_grants=total_grants,
            active_grants=active_grants,
            total_units=total_units,
            used_units=used_units,
            remaining_units=max(0, total_units - used_units),
        ))
    return out


@router.get(
    "/services/me",
    response_model=list[MyServiceOut],
)
async def my_services(
    auth: tuple[ServiceInstance | None, InstanceApiKey] = Depends(
        verify_bearer_token_any,
    ),
    session: AsyncSession = Depends(get_async_session),
):
    """Return every service this bearer-token key can reach, with a
    quota summary. Legacy 1:1 keys see a single row for their bound
    instance (no grant/pack info unless explicitly provisioned).
    """
    instance, api_key = auth

    rows: list[MyServiceOut] = []

    if instance is not None:
        # Legacy key: one row, no grant/pack info.
        rows.append(MyServiceOut(
            instance_id=instance.id, instance_name=instance.name,
            category=instance.category, meter_dim=instance.meter_dim,
            grant_status="legacy",
            total_units=0, used_units=0, remaining_units=0,
        ))
        return rows

    # M:N: join grants → instances, sum packs per grant.
    grants_stmt = (
        select(ApiKeyGrant, ServiceInstance)
        .join(ServiceInstance, ServiceInstance.id == ApiKeyGrant.service_id)
        .where(ApiKeyGrant.api_key_id == api_key.id)
    )
    for grant, inst in (await session.execute(grants_stmt)).all():
        packs = (await session.execute(
            select(ResourcePack).where(ResourcePack.grant_id == grant.id)
        )).scalars().all()
        total = sum(p.total_units for p in packs)
        used = sum(p.used_units for p in packs)
        rows.append(MyServiceOut(
            instance_id=inst.id, instance_name=inst.name,
            category=inst.category, meter_dim=inst.meter_dim,
            grant_status=grant.status,
            total_units=total, used_units=used,
            remaining_units=max(0, total - used),
        ))
    return rows
