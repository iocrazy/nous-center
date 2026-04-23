"""v3 model_resolver tests.

The v3 resolver maps (api_key, request.model) → ServiceInstance through
ApiKeyGrant lookups only. Legacy 1:1 binding asserts were removed in PR-A
because the v3 migration NULLs out `instance_api_keys.instance_id` and
the resolver no longer special-cases it.
"""

from __future__ import annotations

import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.model_resolver import (
    ModelNotFound,
    resolve_target_service,
)


async def _make_service(session, *, name: str):
    svc = ServiceInstance(
        source_type="model", source_name=name, name=name, type="llm",
        category="llm", meter_dim="tokens",
    )
    session.add(svc)
    await session.commit()
    await session.refresh(svc)
    return svc


async def _make_key(session, *, prefix: str = "sk-x"):
    key = InstanceApiKey(
        instance_id=None, label="k", key_hash="h",
        key_prefix=prefix,
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)
    return key


@pytest.mark.asyncio
async def test_grant_match_by_service_name(db_session):
    """Active grant + matching name resolves to the right service."""
    svc_a = await _make_service(db_session, name="qwen3")
    svc_b = await _make_service(db_session, name="deepseek")
    key = await _make_key(db_session, prefix="sk-c")
    db_session.add_all([
        ApiKeyGrant(api_key_id=key.id, service_id=svc_a.id, status="active"),
        ApiKeyGrant(api_key_id=key.id, service_id=svc_b.id, status="active"),
    ])
    await db_session.commit()

    resolved = await resolve_target_service(
        db_session, api_key=key, requested_model="deepseek",
    )
    assert resolved.id == svc_b.id


@pytest.mark.asyncio
async def test_grant_unknown_service_raises(db_session):
    svc = await _make_service(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-d")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=svc.id))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_service(
            db_session, api_key=key, requested_model="ghost-model",
        )


@pytest.mark.asyncio
async def test_grant_missing_model_name_raises(db_session):
    """No request.model → can't pick a service, must fail (no legacy default)."""
    svc = await _make_service(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-e")
    db_session.add(ApiKeyGrant(api_key_id=key.id, service_id=svc.id))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_service(
            db_session, api_key=key, requested_model=None,
        )


@pytest.mark.asyncio
async def test_paused_grant_is_skipped(db_session):
    """Only active grants are visible to the resolver."""
    svc = await _make_service(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-f")
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, service_id=svc.id, status="paused",
    ))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_service(
            db_session, api_key=key, requested_model="qwen3",
        )


@pytest.mark.asyncio
async def test_no_grants_raises(db_session):
    key = await _make_key(db_session, prefix="sk-g")
    with pytest.raises(ModelNotFound):
        await resolve_target_service(
            db_session, api_key=key, requested_model="qwen3",
        )
