"""Quota enforcement for the inference protocol layer.

Callers wrap each usage-recording path with `consume_for_request(...)`.
The function is deliberately small: it finds the active grant, charges
it, and fires any alerts. Nothing else.

Why alert checks swallow errors: a failure in the alert path (bad rule,
DB hiccup, misconfigured threshold) must never turn a successful
inference into a user-visible 500. The quota consume already committed
before we get here, so we log and move on.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import func

from src.models.api_gateway import ApiKeyGrant, ResourcePack
from src.services.alert_rule import AlertEvent, check_and_fire
from src.services.resource_pack import ConsumeResult, QuotaExhausted, consume, peek_remaining

logger = logging.getLogger(__name__)


class NoActiveGrant(Exception):
    """The key has no active grant for this service — 403-shaped, not 402."""


async def preflight_check(
    session: AsyncSession,
    *,
    api_key_id: int,
    service_id: int,
) -> None:
    """推理**前**门:该 key 对该 service 有 active grant 但余量 <= 0 → 抛 QuotaExhausted(402)。

    安全 review P2:原先配额只在推理**后**结算且吞掉 QuotaExhausted → 已耗尽的 key 靠
    并发/突发可无限超额刷 GPU(每个请求都先跑完才发现超额)。这里在昂贵推理前先挡一道。

    - 无 grant(legacy 1:1 key)→ 放行,与 post-consume 计费侧一致(那边 NoActiveGrant 静默跳过)。
    - 有 grant 但**没有任何 ResourcePack** → 放行:这是"已授权但未配额度"= 无限量,不是耗尽
      (consume 对无 pack 的 grant 本就 no-op/放行语义)。
    - 有 grant 且有 pack 但可用余量 <= 0 → QuotaExhausted。
    注意 peek + 真正 consume 非原子,并发窗口仍存在,但已把"已耗尽 key 无限跑"收敛为
    "至多再跑 ~并发数 个请求";consume() 的原子 CAS 仍是最终真值。
    """
    grant_id = await session.scalar(
        select(ApiKeyGrant.id).where(
            ApiKeyGrant.api_key_id == api_key_id,
            ApiKeyGrant.service_id == service_id,
            ApiKeyGrant.status == "active",
        )
    )
    if grant_id is None:
        return  # legacy / 无 grant —— 不拦(计费侧也跳过)
    pack_count = await session.scalar(
        select(func.count(ResourcePack.id)).where(ResourcePack.grant_id == grant_id)
    )
    if not pack_count:
        return  # 有 grant 但未配任何额度包 = 无限量,放行
    remaining = await peek_remaining(session, grant_id=grant_id)
    if remaining <= 0:
        raise QuotaExhausted(
            f"grant {grant_id} for service {service_id} has no remaining units",
        )


async def consume_for_request(
    session: AsyncSession,
    *,
    api_key_id: int,
    service_id: int,
    units: int,
    allow_overshoot: bool = False,
) -> tuple[ConsumeResult, list[AlertEvent]]:
    """Atomically charge `units` against the grant for (api_key, service).

    Returns (ConsumeResult, events). Raises:
      - NoActiveGrant: caller is not authorized; return 403.
      - QuotaExhausted (from resource_pack.consume): return 402.

    allow_overshoot(H1):post-work 结算传 True —— 工作已交付,额度耗尽也强制记账
    (扣成负),不漏计。仅无 pack(无限量)grant 仍抛 QuotaExhausted。见 resource_pack.consume。
    """
    grant_id = await session.scalar(
        select(ApiKeyGrant.id).where(
            ApiKeyGrant.api_key_id == api_key_id,
            ApiKeyGrant.service_id == service_id,
            ApiKeyGrant.status == "active",
        )
    )
    if grant_id is None:
        raise NoActiveGrant(
            f"api_key {api_key_id} has no active grant on service {service_id}",
        )

    result = await consume(
        session, grant_id=grant_id, units=units, allow_overshoot=allow_overshoot,
    )

    # Best-effort: alert failures must not block the caller.
    try:
        events = await check_and_fire(session, grant_id=grant_id)
    except Exception:
        logger.exception("alert check failed for grant %s", grant_id)
        events = []

    return result, events
