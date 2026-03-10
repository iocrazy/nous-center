from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import (
    VoicePresetCreate,
    VoicePresetGroupCreate,
    VoicePresetGroupOut,
    VoicePresetOut,
    VoicePresetUpdate,
)
from src.models.voice_preset import VoicePreset, VoicePresetGroup

router = APIRouter(prefix="/api/v1/voices", tags=["voices"])


@router.get("", response_model=list[VoicePresetOut])
async def list_presets(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(VoicePreset).order_by(VoicePreset.created_at))
    return result.scalars().all()


@router.post("", response_model=VoicePresetOut, status_code=201)
async def create_preset(
    data: VoicePresetCreate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = VoicePreset(**data.model_dump())
    session.add(preset)
    await session.commit()
    await session.refresh(preset)
    return preset


# --- Voice Preset Groups (before /{preset_id} to avoid route conflict) ---


@router.get("/groups", response_model=list[VoicePresetGroupOut])
async def list_groups(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(VoicePresetGroup).order_by(VoicePresetGroup.created_at))
    return result.scalars().all()


@router.post("/groups", response_model=VoicePresetGroupOut, status_code=201)
async def create_group(
    data: VoicePresetGroupCreate,
    session: AsyncSession = Depends(get_async_session),
):
    group = VoicePresetGroup(**data.model_dump())
    session.add(group)
    await session.commit()
    await session.refresh(group)
    return group


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(
    group_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    group = await session.get(VoicePresetGroup, group_id)
    if not group:
        raise HTTPException(404, detail="Voice preset group not found")
    await session.delete(group)
    await session.commit()


# --- Individual Preset CRUD ---


@router.get("/{preset_id}", response_model=VoicePresetOut)
async def get_preset(
    preset_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    return preset


@router.put("/{preset_id}", response_model=VoicePresetOut)
async def update_preset(
    preset_id: UUID,
    data: VoicePresetUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(preset, key, value)
    await session.commit()
    await session.refresh(preset)
    return preset


@router.delete("/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Voice preset not found")
    await session.delete(preset)
    await session.commit()
