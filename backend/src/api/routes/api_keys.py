"""m10 全局 API Key 管理 — key-centric 视角。

老的 instance_keys.py 是 instance-scoped（"给某 service 加 key"），
本模块是 key-scoped（"创建 key，授权到 N 个服务"），匹配 v3 IA 把
API Key 提为一等公民的设计：

  POST   /api/v1/keys                 创建 key（可附 service_ids 一键授权）
  GET    /api/v1/keys                 全局列表（带 grant_count + 用量摘要）
  GET    /api/v1/keys/{id}            详情（含 grants/services）
  PATCH  /api/v1/keys/{id}            改 label/note/expires_at/is_active
  DELETE /api/v1/keys/{id}
  POST   /api/v1/keys/{id}/reset      重置 secret（返回新明文）

明文 key 的存储：参考阿里百炼模式 — 创建/重置时同时写 key_hash 和
secret_plaintext。bcrypt 验证仍走 key_hash（不变），UI 用 plaintext
让管理员随时复看而无需"创建一次错过即丢"。

授权 grant 的增删走老的 /api/v1/keys/{id}/grants（api_gateway.py），
本模块只在创建时支持一次性多 grant。
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_serializer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.models.api_gateway import ApiKeyGrant
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

router = APIRouter(
    prefix="/api/v1/keys",
    tags=["api-keys"],
    dependencies=[Depends(require_admin)],
)


# ---------- Pydantic ----------


class KeyCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    note: str | None = None
    expires_at: datetime | None = None
    # 一键多授权：创建 key 同时把它授权到这些 service。
    service_ids: list[int] = Field(default_factory=list)


class KeyPatch(BaseModel):
    label: str | None = Field(None, min_length=1, max_length=100)
    note: str | None = None
    expires_at: datetime | None = None
    is_active: bool | None = None


class GrantSummary(BaseModel):
    id: int
    service_id: int
    service_name: str
    service_category: str | None
    status: str
    activated_at: datetime

    # Snowflake-style 64-bit IDs overflow JS Number. Serialize as string in
    # JSON so the frontend can pass the value back round-trip without losing
    # precision — otherwise PATCH/DELETE /grants/{id} 404s on the wrong id.
    @field_serializer("id", "service_id", when_used="json")
    def _id_to_str(self, v: int) -> str:
        return str(v)


class KeyOut(BaseModel):
    id: int
    label: str
    note: str | None
    key_prefix: str
    secret_plaintext: str | None  # null = legacy/rotated 之前的 key
    is_active: bool
    usage_calls: int
    usage_chars: int
    last_used_at: datetime | None
    created_at: datetime | None
    expires_at: datetime | None
    grant_count: int
    active_grant_count: int
    grants: list[GrantSummary] = Field(default_factory=list)

    @field_serializer("id", when_used="json")
    def _id_to_str(self, v: int) -> str:
        return str(v)


class KeyCreated(KeyOut):
    """创建/reset 返回。secret 字段是真正的明文 — 调用方必须保存。"""
    secret: str


# ---------- helpers ----------


def _gen_secret(label: str) -> tuple[str, str, str]:
    """Returns (full_key, bcrypt_hash, prefix)."""
    clean = re.sub(r"[^a-zA-Z0-9]", "", label)[:4].lower() or "key"
    rnd = os.urandom(16).hex()
    full = f"sk-{clean}-{rnd}"
    return full, bcrypt.hashpw(full.encode(), bcrypt.gensalt()).decode(), full[:10]


async def _grant_summaries(
    session: AsyncSession, key_id: int,
) -> list[GrantSummary]:
    # LEFT JOIN：v3 IA 重构清理过 service_instances，旧 grant 仍指向已删
    # 的 service_id。INNER JOIN 会让这些孤儿 grant 整行丢失，导致 UI
    # 看到 grant_count > 0 但 grants: []，没法解除。LEFT JOIN + 兜底
    # service_name="(已删除)" 让用户能从详情页清掉孤儿。
    rows = (await session.execute(
        select(ApiKeyGrant, ServiceInstance)
        .outerjoin(ServiceInstance, ServiceInstance.id == ApiKeyGrant.service_id)
        .where(ApiKeyGrant.api_key_id == key_id)
        .order_by(ApiKeyGrant.activated_at.desc())
    )).all()
    return [
        GrantSummary(
            id=g.id,
            service_id=g.service_id,
            service_name=svc.name if svc else "(已删除)",
            service_category=svc.category if svc else None,
            status=g.status,
            activated_at=g.activated_at,
        )
        for g, svc in rows
    ]


def _to_out(
    key: InstanceApiKey,
    grant_count: int,
    active_count: int,
    grants: list[GrantSummary] | None = None,
) -> KeyOut:
    return KeyOut(
        id=key.id, label=key.label, note=key.note,
        key_prefix=key.key_prefix, secret_plaintext=key.secret_plaintext,
        is_active=key.is_active, usage_calls=key.usage_calls,
        usage_chars=key.usage_chars, last_used_at=key.last_used_at,
        created_at=key.created_at, expires_at=key.expires_at,
        grant_count=grant_count, active_grant_count=active_count,
        grants=grants or [],
    )


# ---------- routes ----------


@router.post("", response_model=KeyCreated, status_code=201)
async def create_key(
    body: KeyCreate,
    session: AsyncSession = Depends(get_async_session),
):
    full, key_hash, prefix = _gen_secret(body.label)

    key = InstanceApiKey(
        instance_id=None,  # M:N — 走 grants
        label=body.label,
        key_hash=key_hash,
        key_prefix=prefix,
        secret_plaintext=full,
        note=body.note,
        expires_at=body.expires_at,
    )
    session.add(key)
    await session.flush()

    # 一键多授权：先验证全部 service_id 都存在 + 去重。
    svc_ids = list(dict.fromkeys(body.service_ids))
    if svc_ids:
        existing_svcs = (await session.execute(
            select(ServiceInstance.id).where(ServiceInstance.id.in_(svc_ids))
        )).scalars().all()
        missing = set(svc_ids) - set(existing_svcs)
        if missing:
            raise HTTPException(
                404, detail=f"service(s) not found: {sorted(missing)}",
            )
        for sid in svc_ids:
            session.add(ApiKeyGrant(api_key_id=key.id, service_id=sid))

    await session.commit()
    await session.refresh(key)

    grants = await _grant_summaries(session, key.id)
    out = _to_out(
        key,
        grant_count=len(grants),
        active_count=sum(1 for g in grants if g.status == "active"),
        grants=grants,
    )
    return KeyCreated(secret=full, **out.model_dump())


@router.get("", response_model=list[KeyOut])
async def list_keys(
    session: AsyncSession = Depends(get_async_session),
):
    keys = (await session.execute(
        select(InstanceApiKey).order_by(InstanceApiKey.created_at.desc())
    )).scalars().all()

    # 跨 PG/SQLite 兼容：拆成两条 GROUP BY，避开 boolean→int 的 cast 差异
    # （和 api_gateway.services_catalog 同模式）。
    totals = {
        kid: int(c)
        for kid, c in (await session.execute(
            select(ApiKeyGrant.api_key_id, func.count(ApiKeyGrant.id))
            .group_by(ApiKeyGrant.api_key_id)
        )).all()
    }
    actives = {
        kid: int(c)
        for kid, c in (await session.execute(
            select(ApiKeyGrant.api_key_id, func.count(ApiKeyGrant.id))
            .where(ApiKeyGrant.status == "active")
            .group_by(ApiKeyGrant.api_key_id)
        )).all()
    }

    # 一次取齐所有 key 的 grants（含孤儿），按 api_key_id 分组。m10 列表
    # 直接用这个数组渲染"授权服务"徽章，不必再调 N 次 detail。
    grants_by_key: dict[int, list[GrantSummary]] = {}
    for g, svc in (await session.execute(
        select(ApiKeyGrant, ServiceInstance)
        .outerjoin(ServiceInstance, ServiceInstance.id == ApiKeyGrant.service_id)
        .order_by(ApiKeyGrant.activated_at.desc())
    )).all():
        grants_by_key.setdefault(g.api_key_id, []).append(
            GrantSummary(
                id=g.id,
                service_id=g.service_id,
                service_name=svc.name if svc else "(已删除)",
                service_category=svc.category if svc else None,
                status=g.status,
                activated_at=g.activated_at,
            )
        )

    return [
        _to_out(
            k,
            grant_count=totals.get(k.id, 0),
            active_count=actives.get(k.id, 0),
            grants=grants_by_key.get(k.id, []),
        )
        for k in keys
    ]


@router.get("/{key_id}", response_model=KeyOut)
async def get_key(
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    key = await session.get(InstanceApiKey, key_id)
    if not key:
        raise HTTPException(404, detail="api key not found")
    grants = await _grant_summaries(session, key_id)
    return _to_out(
        key,
        grant_count=len(grants),
        active_count=sum(1 for g in grants if g.status == "active"),
        grants=grants,
    )


@router.patch("/{key_id}", response_model=KeyOut)
async def patch_key(
    key_id: int,
    body: KeyPatch,
    session: AsyncSession = Depends(get_async_session),
):
    key = await session.get(InstanceApiKey, key_id)
    if not key:
        raise HTTPException(404, detail="api key not found")
    if body.label is not None:
        key.label = body.label
    if body.note is not None:
        key.note = body.note
    if body.expires_at is not None:
        key.expires_at = body.expires_at
    if body.is_active is not None:
        key.is_active = body.is_active
    await session.commit()
    await session.refresh(key)
    grants = await _grant_summaries(session, key_id)
    return _to_out(
        key,
        grant_count=len(grants),
        active_count=sum(1 for g in grants if g.status == "active"),
        grants=grants,
    )


@router.post("/{key_id}/reset", response_model=KeyCreated)
async def reset_key(
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """轮换 secret。旧 secret 立即失效（hash 被覆盖）。grants 不变。"""
    key = await session.get(InstanceApiKey, key_id)
    if not key:
        raise HTTPException(404, detail="api key not found")
    full, key_hash, prefix = _gen_secret(key.label)
    key.key_hash = key_hash
    key.key_prefix = prefix
    key.secret_plaintext = full
    # reset 视为"重新启用"：把 last_used_at 留着方便审计。
    await session.commit()
    await session.refresh(key)
    grants = await _grant_summaries(session, key_id)
    out = _to_out(
        key,
        grant_count=len(grants),
        active_count=sum(1 for g in grants if g.status == "active"),
        grants=grants,
    )
    return KeyCreated(secret=full, **out.model_dump())


@router.delete("/{key_id}", status_code=204)
async def delete_key(
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    key = await session.get(InstanceApiKey, key_id)
    if not key:
        raise HTTPException(404, detail="api key not found")
    # 显式删 grants（SQLite 默认不强制 FK cascade，PG 强制；
    # 两边一致最省心）。
    grants = (await session.execute(
        select(ApiKeyGrant).where(ApiKeyGrant.api_key_id == key_id)
    )).scalars().all()
    for g in grants:
        await session.delete(g)
    await session.delete(key)
    await session.commit()


# ---------- 给 m03 ServiceDetail 的 "Key 授权" tab 用 ----------


class ServiceKeyGrantOut(BaseModel):
    grant_id: int
    api_key_id: int
    api_key_label: str
    api_key_prefix: str
    grant_status: str
    activated_at: datetime
    pack_total: int
    pack_used: int

    @field_serializer("grant_id", "api_key_id", when_used="json")
    def _id_to_str(self, v: int) -> str:
        return str(v)


service_grants_router = APIRouter(
    prefix="/api/v1/services",
    tags=["api-keys"],
    dependencies=[Depends(require_admin)],
)


@service_grants_router.get(
    "/{service_id}/grants",
    response_model=list[ServiceKeyGrantOut],
)
async def list_service_grants(
    service_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """某个 service 上的所有 key grant + 配额聚合。m03 "Key 授权" tab 用。"""
    from src.models.api_gateway import ResourcePack

    if not await session.get(ServiceInstance, service_id):
        raise HTTPException(404, detail="service not found")

    rows = (await session.execute(
        select(ApiKeyGrant, InstanceApiKey)
        .join(InstanceApiKey, InstanceApiKey.id == ApiKeyGrant.api_key_id)
        .where(ApiKeyGrant.service_id == service_id)
        .order_by(ApiKeyGrant.activated_at.desc())
    )).all()

    pack_sums = {
        gid: (int(total or 0), int(used or 0))
        for gid, total, used in (await session.execute(
            select(
                ResourcePack.grant_id,
                func.coalesce(func.sum(ResourcePack.total_units), 0),
                func.coalesce(func.sum(ResourcePack.used_units), 0),
            ).group_by(ResourcePack.grant_id)
        )).all()
    }

    return [
        ServiceKeyGrantOut(
            grant_id=g.id,
            api_key_id=k.id,
            api_key_label=k.label,
            api_key_prefix=k.key_prefix,
            grant_status=g.status,
            activated_at=g.activated_at,
            pack_total=pack_sums.get(g.id, (0, 0))[0],
            pack_used=pack_sums.get(g.id, (0, 0))[1],
        )
        for g, k in rows
    ]
