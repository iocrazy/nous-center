"""Service instance CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.service_instance import ServiceInstance
from src.models.voice_preset import VoicePreset
from src.models.workflow import Workflow
from src.models.schemas import (
    ServiceInstanceCreate,
    ServiceInstanceOut,
    ServiceInstanceUpdate,
    InstanceStatusUpdate,
)

router = APIRouter(prefix="/api/v1/instances", tags=["instances"])


async def _resolve_source_name(instance: ServiceInstance, session: AsyncSession) -> str | None:
    """Resolve a human-readable name for the instance source."""
    if instance.source_type == "model":
        return instance.source_name
    if instance.source_type == "preset" and instance.source_id:
        preset = await session.get(VoicePreset, instance.source_id)
        return preset.name if preset else None
    if instance.source_type == "workflow" and instance.source_id:
        wf = await session.get(Workflow, instance.source_id)
        return wf.name if wf else None
    return None


def _instance_to_out(instance: ServiceInstance, source_name: str | None = None) -> dict:
    """Convert instance + resolved source_name to dict for ServiceInstanceOut."""
    return {
        "id": instance.id,
        "source_type": instance.source_type,
        "source_id": instance.source_id,
        "source_name": source_name,
        "name": instance.name,
        "type": instance.type,
        "status": instance.status,
        "endpoint_path": instance.endpoint_path,
        "params_override": instance.params_override,
        "created_at": instance.created_at,
        "updated_at": instance.updated_at,
    }


@router.post("", response_model=ServiceInstanceOut, status_code=201)
async def create_instance(
    data: ServiceInstanceCreate,
    session: AsyncSession = Depends(get_async_session),
):
    # Validate source exists
    if data.source_type == "preset":
        preset = await session.get(VoicePreset, data.source_id)
        if not preset:
            raise HTTPException(404, detail="Source preset not found")
        source_name = preset.name
    elif data.source_type == "workflow":
        wf = await session.get(Workflow, data.source_id)
        if not wf:
            raise HTTPException(404, detail="Source workflow not found")
        source_name = wf.name
    elif data.source_type == "model":
        source_name = data.source_name
        if not source_name:
            raise HTTPException(400, detail="source_name required for model type")
    else:
        raise HTTPException(400, detail=f"Unsupported source_type: {data.source_type}")

    instance = ServiceInstance(
        source_type=data.source_type,
        source_id=data.source_id,
        source_name=source_name if data.source_type == "model" else None,
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

    return _instance_to_out(instance, source_name)


@router.get("", response_model=list[ServiceInstanceOut])
async def list_instances(
    type: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    query = select(ServiceInstance).order_by(ServiceInstance.created_at.desc())
    if type is not None:
        query = query.where(ServiceInstance.type == type)
    result = await session.execute(query)
    instances = result.scalars().all()

    # Batch-resolve source names to avoid N+1
    preset_ids = {inst.source_id for inst in instances if inst.source_type == "preset"}
    source_names: dict[int, str | None] = {}
    if preset_ids:
        preset_result = await session.execute(
            select(VoicePreset.id, VoicePreset.name).where(VoicePreset.id.in_(preset_ids))
        )
        preset_map = dict(preset_result.all())
        for inst in instances:
            if inst.source_type == "preset":
                source_names[inst.id] = preset_map.get(inst.source_id)

    wf_ids = [i.source_id for i in instances if i.source_type == "workflow"]
    if wf_ids:
        wf_result = await session.execute(
            select(Workflow.id, Workflow.name).where(Workflow.id.in_(wf_ids))
        )
        wf_map = dict(wf_result.all())
        for inst in instances:
            if inst.source_type == "workflow":
                source_names[inst.id] = wf_map.get(inst.source_id)

    out = []
    for inst in instances:
        name = source_names.get(inst.id)
        out.append(_instance_to_out(inst, name))
    return out


@router.get("/{instance_id}", response_model=ServiceInstanceOut)
async def get_instance(
    instance_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    instance = await session.get(ServiceInstance, instance_id)
    if not instance:
        raise HTTPException(404, detail="Instance not found")
    source_name = await _resolve_source_name(instance, session)
    return _instance_to_out(instance, source_name)


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
    if data.rate_limit_rpm is not None:
        instance.rate_limit_rpm = data.rate_limit_rpm or None
    if data.rate_limit_tpm is not None:
        instance.rate_limit_tpm = data.rate_limit_tpm or None

    await session.commit()
    await session.refresh(instance)
    source_name = await _resolve_source_name(instance, session)
    return _instance_to_out(instance, source_name)


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
