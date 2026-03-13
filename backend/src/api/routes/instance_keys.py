"""API key management routes for service instances."""

import os
import re

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.models.schemas import (
    InstanceApiKeyCreate,
    InstanceApiKeyCreated,
    InstanceApiKeyOut,
)

router = APIRouter(prefix="/api/v1/instances", tags=["instance-keys"])


def _generate_key(instance_name: str) -> tuple[str, str, str]:
    """Generate API key. Returns (full_key, key_hash, key_prefix)."""
    clean = re.sub(r"[^a-zA-Z0-9]", "", instance_name)[:4].lower() or "key"
    random_hex = os.urandom(16).hex()
    full_key = f"sk-{clean}-{random_hex}"
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
    key_prefix = full_key[:10]
    return full_key, key_hash, key_prefix


@router.post("/{instance_id}/keys", response_model=InstanceApiKeyCreated, status_code=201)
async def create_key(
    instance_id: int,
    data: InstanceApiKeyCreate,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")

    full_key, key_hash, key_prefix = _generate_key(instance.name)

    api_key = InstanceApiKey(
        instance_id=instance_id,
        label=data.label,
        key_hash=key_hash,
        key_prefix=key_prefix,
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)

    # Return full key only in creation response
    data_dict = InstanceApiKeyOut.model_validate(api_key).model_dump()
    data_dict["key"] = full_key
    return InstanceApiKeyCreated(**data_dict)


@router.get("/{instance_id}/keys", response_model=list[InstanceApiKeyOut])
async def list_keys(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")

    result = await session.execute(
        select(InstanceApiKey)
        .where(InstanceApiKey.instance_id == instance_id)
        .order_by(InstanceApiKey.created_at)
    )
    return result.scalars().all()


@router.delete("/{instance_id}/keys/{key_id}", status_code=204)
async def delete_key(
    instance_id: int,
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    api_key = await session.get(InstanceApiKey, key_id)
    if not api_key or api_key.instance_id != instance_id:
        raise HTTPException(404, detail="API key not found")
    await session.delete(api_key)
    await session.commit()
