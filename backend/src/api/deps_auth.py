"""Bearer Token authentication for instance service endpoints."""

from datetime import datetime, timezone

import bcrypt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.errors import AuthenticationError
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.rate_limiter import get_rate_limiter

# Pre-computed dummy hash for constant-time rejection (prevents timing attacks)
_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt()).decode()


def _key_expired(key: InstanceApiKey) -> bool:
    if key.expires_at is None:
        return False
    exp = key.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp < datetime.now(timezone.utc)


async def enforce_instance_rate_limit(instance: ServiceInstance) -> None:
    """准入即占坑 —— 鉴权通过(或 M:N key 解析出目标 instance)后立刻调一次。

    M:N key 走 `verify_bearer_token_any` 返回 (None, key),限流必须由调用方在解析出
    instance 之后补调本函数(openai/anthropic/ollama compat),否则那条路径不限流。
    """
    await get_rate_limiter().reserve(
        instance.id,
        getattr(instance, "rate_limit_rpm", None),
        getattr(instance, "rate_limit_tpm", None),
    )


async def verify_bearer_token_any(
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey]:
    """Global bearer auth for /v1/* endpoints, tolerant of M:N keys (instance_id NULL).

    Returned instance is None for M:N keys; the caller resolves the target via
    `model_resolver.resolve_target_service(api_key, request.model)` then enforces
    rate limits on it.

    Legacy keys (instance_id set) are resolved and rate-limited up-front. No live
    subsystem creates such keys anymore (the instance/instance_keys 1:1 routes were
    deleted), but the branch + test fixtures still exercise it; collapsing
    verify_bearer_token_any to M:N-only is a separate PR that must first migrate the
    conftest default key (legacy → M:N + grant) and re-verify the full suite.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Invalid authorization header")
    token = authorization[7:]

    key_prefix = token[:10]
    result = await session.execute(
        select(InstanceApiKey).where(
            InstanceApiKey.key_prefix == key_prefix,
            InstanceApiKey.is_active == True,  # noqa: E712
        )
    )
    keys = result.scalars().all()

    for key in keys:
        if bcrypt.checkpw(token.encode(), key.key_hash.encode()):
            if _key_expired(key):
                raise AuthenticationError(
                    "API key expired", code="api_key_expired",
                )
            if key.instance_id is None:
                # M:N key; rate limits enforced after resolution.
                return None, key
            instance = await session.get(ServiceInstance, key.instance_id)
            if instance and instance.status == "active":
                await enforce_instance_rate_limit(instance)
                return instance, key
            raise HTTPException(403, detail="Instance is inactive")

    bcrypt.checkpw(token.encode(), _DUMMY_HASH.encode())
    raise HTTPException(401, detail="Invalid API key")
