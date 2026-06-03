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
    """Global bearer auth for /v1/* endpoints (M:N keys only).

    Always returns ``(None, key)``; the caller resolves the target service via
    `model_resolver.resolve_target_service(api_key, request.model)` and enforces
    rate limits on it. The legacy 1:1 binding regime is gone (the instance /
    instance_keys / verify_instance_key subsystem was deleted in the legacy rip);
    `InstanceApiKey.instance_id` stays nullable for schema stability but is always
    NULL. The ``(instance, key)`` tuple shape is kept so callers stay uniform.
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
            # M:N: rate limits enforced by the caller after model resolution.
            return None, key

    # Always do one bcrypt round to prevent timing-based probing.
    bcrypt.checkpw(token.encode(), _DUMMY_HASH.encode())
    raise HTTPException(401, detail="Invalid API key")
