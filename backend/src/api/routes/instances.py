"""Service instance CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.service_instance import ServiceInstance
from src.models.voice_preset import VoicePreset
from src.models.schemas import (
    ServiceInstanceCreate,
    ServiceInstanceOut,
    ServiceInstanceUpdate,
    InstanceStatusUpdate,
)

router = APIRouter(prefix="/api/v1/instances", tags=["instances"])


@router.post("", response_model=ServiceInstanceOut, status_code=201)
async def create_instance(
    data: ServiceInstanceCreate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, data.preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")

    instance = ServiceInstance(
        preset_id=data.preset_id,
        name=data.name,
        type=data.type,
        params_override=data.params_override,
    )
    session.add(instance)
    await session.commit()
    await session.refresh(instance)

    # Auto-set endpoint_path
    instance.endpoint_path = f"/v1/instances/{instance.id}/synthesize"
    await session.commit()
    await session.refresh(instance)

    return instance


@router.get("", response_model=list[ServiceInstanceOut])
async def list_instances(
    preset_id: int | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    query = select(ServiceInstance).order_by(ServiceInstance.created_at.desc())
    if preset_id is not None:
        query = query.where(ServiceInstance.preset_id == preset_id)
    result = await session.execute(query)
    return result.scalars().all()


@router.get("/{instance_id}", response_model=ServiceInstanceOut)
async def get_instance(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    return instance


@router.patch("/{instance_id}", response_model=ServiceInstanceOut)
async def update_instance(
    instance_id: int,
    data: ServiceInstanceUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")

    if data.name is not None:
        instance.name = data.name
    if data.params_override is not None:
        instance.params_override = data.params_override

    await session.commit()
    await session.refresh(instance)
    return instance


@router.patch("/{instance_id}/status")
async def update_instance_status(
    instance_id: int,
    data: InstanceStatusUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    instance.status = data.status
    await session.commit()
    await session.refresh(instance)
    return {"id": instance.id, "status": instance.status}


@router.delete("/{instance_id}", status_code=204)
async def delete_instance(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    await session.delete(instance)
    await session.commit()
