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


async def _enforce_instance_limits(instance: ServiceInstance) -> None:
    await get_rate_limiter().check(
        instance.id,
        getattr(instance, "rate_limit_rpm", None),
        getattr(instance, "rate_limit_tpm", None),
    )


async def verify_instance_key(
    instance_id: int,
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance, InstanceApiKey]:
    """Verify Bearer token against instance API keys.

    Returns (instance, matched_key) on success.
    Raises 401/403/404 on failure.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Invalid authorization header")
    token = authorization[7:]

    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    if instance.status != "active":
        raise HTTPException(403, detail="Instance is inactive")

    # Use key_prefix to narrow candidates before bcrypt (O(1) instead of O(n))
    key_prefix = token[:10]
    result = await session.execute(
        select(InstanceApiKey).where(
            InstanceApiKey.instance_id == instance_id,
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
            await _enforce_instance_limits(instance)
            return instance, key

    # Always do one bcrypt round to prevent timing-based probing
    bcrypt.checkpw(token.encode(), _DUMMY_HASH.encode())

    raise HTTPException(401, detail="Invalid API key")


async def verify_bearer_token(
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance, InstanceApiKey]:
    """Verify Bearer token without requiring instance_id in URL.

    Looks up the API key by prefix across all active instances.
    Used for /v1/chat/completions where the instance is implicit.
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
            instance = await session.get(ServiceInstance, key.instance_id)
            if instance and instance.status == "active":
                if _key_expired(key):
                    raise AuthenticationError(
                        "API key expired", code="api_key_expired",
                    )
                await _enforce_instance_limits(instance)
                return instance, key
            raise HTTPException(403, detail="Instance is inactive")

    bcrypt.checkpw(token.encode(), _DUMMY_HASH.encode())
    raise HTTPException(401, detail="Invalid API key")


async def verify_bearer_token_any(
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[ServiceInstance | None, InstanceApiKey]:
    """Like verify_bearer_token but tolerates M:N keys (instance_id NULL).

    Returned instance is None for M:N keys; the caller is expected to use
    `model_resolver.resolve_target_instance(api_key, request.model)` to
    pick the target ServiceInstance, then enforce rate limits on it.

    Legacy keys (instance_id set) behave exactly as verify_bearer_token:
    instance is resolved and rate-limited up-front.
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
                await _enforce_instance_limits(instance)
                return instance, key
            raise HTTPException(403, detail="Instance is inactive")

    bcrypt.checkpw(token.encode(), _DUMMY_HASH.encode())
    raise HTTPException(401, detail="Invalid API key")
