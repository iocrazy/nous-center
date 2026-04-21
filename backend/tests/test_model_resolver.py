"""Lane B-T1 · model_resolver tests.

The resolver maps (api_key, request.model) → ServiceInstance.

Legacy path: if the api_key row has instance_id set (pre-M:N binding), that
instance wins and request.model is ignored. This preserves the behavior
existing customers already have wired up.

M:N path: api_key.instance_id is NULL → look up ApiKeyGrant rows for the
key and match request.model against ServiceInstance.name. Status must be
"active".
"""

from __future__ import annotations

import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.services.model_resolver import (
    ModelNotFound,
    resolve_target_instance,
)


async def _make_instance(session, *, name: str, source_type: str = "model"):
    inst = ServiceInstance(
        source_type=source_type, source_name=name, name=name, type="llm",
        category="llm", meter_dim="tokens",
    )
    session.add(inst)
    await session.commit()
    await session.refresh(inst)
    return inst


async def _make_key(session, *, prefix: str = "sk-x", instance_id=None):
    key = InstanceApiKey(
        instance_id=instance_id, label="k", key_hash="h",
        key_prefix=prefix,
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)
    return key


@pytest.mark.asyncio
async def test_legacy_1to1_binding_wins(db_session):
    """api_key.instance_id set → returns that instance, ignores request.model."""
    inst = await _make_instance(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-a", instance_id=inst.id)

    # request.model is deliberately a lie; legacy binding still wins.
    resolved = await resolve_target_instance(
        db_session, api_key=key, requested_model="something-else",
    )
    assert resolved.id == inst.id


@pytest.mark.asyncio
async def test_legacy_binding_ignores_empty_model(db_session):
    """Legacy clients don't send request.model sometimes; still works."""
    inst = await _make_instance(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-b", instance_id=inst.id)

    resolved = await resolve_target_instance(
        db_session, api_key=key, requested_model=None,
    )
    assert resolved.id == inst.id


@pytest.mark.asyncio
async def test_grant_match_by_instance_name(db_session):
    """api_key with no legacy binding + an active grant → lookup by name."""
    inst_a = await _make_instance(db_session, name="qwen3")
    inst_b = await _make_instance(db_session, name="deepseek")
    key = await _make_key(db_session, prefix="sk-c", instance_id=None)
    db_session.add_all([
        ApiKeyGrant(api_key_id=key.id, instance_id=inst_a.id, status="active"),
        ApiKeyGrant(api_key_id=key.id, instance_id=inst_b.id, status="active"),
    ])
    await db_session.commit()

    resolved = await resolve_target_instance(
        db_session, api_key=key, requested_model="deepseek",
    )
    assert resolved.id == inst_b.id


@pytest.mark.asyncio
async def test_grant_unknown_model_raises(db_session):
    inst = await _make_instance(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-d", instance_id=None)
    db_session.add(ApiKeyGrant(api_key_id=key.id, instance_id=inst.id))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_instance(
            db_session, api_key=key, requested_model="ghost-model",
        )


@pytest.mark.asyncio
async def test_grant_missing_model_name_raises(db_session):
    """No legacy binding + request.model empty → can't pick, must fail."""
    inst = await _make_instance(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-e", instance_id=None)
    db_session.add(ApiKeyGrant(api_key_id=key.id, instance_id=inst.id))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_instance(
            db_session, api_key=key, requested_model=None,
        )


@pytest.mark.asyncio
async def test_paused_grant_is_skipped(db_session):
    """Only active grants are visible to the resolver."""
    inst = await _make_instance(db_session, name="qwen3")
    key = await _make_key(db_session, prefix="sk-f", instance_id=None)
    db_session.add(ApiKeyGrant(
        api_key_id=key.id, instance_id=inst.id, status="paused",
    ))
    await db_session.commit()

    with pytest.raises(ModelNotFound):
        await resolve_target_instance(
            db_session, api_key=key, requested_model="qwen3",
        )


@pytest.mark.asyncio
async def test_no_grants_no_legacy_raises(db_session):
    key = await _make_key(db_session, prefix="sk-g", instance_id=None)
    with pytest.raises(ModelNotFound):
        await resolve_target_instance(
            db_session, api_key=key, requested_model="qwen3",
        )
