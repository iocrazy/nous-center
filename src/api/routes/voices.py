from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import VoicePresetCreate, VoicePresetUpdate, VoicePresetOut
from src.models.voice_preset import VoicePreset

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
