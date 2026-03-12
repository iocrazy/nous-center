"""Internal key management routes for presets (no auth required)."""

import os
import re

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.preset_api_key import PresetApiKey
from src.models.schemas import (
    PresetApiKeyCreate,
    PresetApiKeyCreated,
    PresetApiKeyOut,
    PresetStatusUpdate,
)
from src.models.voice_preset import VoicePreset

router = APIRouter(prefix="/api/v1/presets", tags=["preset-keys"])


def _generate_key(preset_name: str) -> tuple[str, str, str]:
    """Generate API key. Returns (full_key, key_hash, key_prefix)."""
    # Derive short prefix from preset name (ASCII chars only, fallback to 'key')
    clean = re.sub(r"[^a-zA-Z0-9]", "", preset_name)[:4].lower() or "key"
    random_hex = os.urandom(16).hex()
    full_key = f"sk-{clean}-{random_hex}"
    key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
    key_prefix = full_key[:10]
    return full_key, key_hash, key_prefix


@router.post("/{preset_id}/keys", response_model=PresetApiKeyCreated, status_code=201)
async def create_key(
    preset_id: int,
    data: PresetApiKeyCreate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")

    full_key, key_hash, key_prefix = _generate_key(preset.name)

    api_key = PresetApiKey(
        preset_id=preset_id,
        label=data.label,
        key_hash=key_hash,
        key_prefix=key_prefix,
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)

    # Auto-set endpoint_path if not already set
    if not preset.endpoint_path:
        preset.endpoint_path = f"/v1/preset/{preset.id}/synthesize"
        await session.commit()

    # Return full key only in creation response
    data_dict = PresetApiKeyOut.model_validate(api_key).model_dump()
    data_dict["key"] = full_key
    return PresetApiKeyCreated(**data_dict)


@router.get("/{preset_id}/keys", response_model=list[PresetApiKeyOut])
async def list_keys(
    preset_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")

    result = await session.execute(
        select(PresetApiKey)
        .where(PresetApiKey.preset_id == preset_id)
        .order_by(PresetApiKey.created_at)
    )
    return result.scalars().all()


@router.delete("/{preset_id}/keys/{key_id}", status_code=204)
async def delete_key(
    preset_id: int,
    key_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    api_key = await session.get(PresetApiKey, key_id)
    if not api_key or api_key.preset_id != preset_id:
        raise HTTPException(404, detail="API key not found")
    await session.delete(api_key)
    await session.commit()


@router.patch("/{preset_id}/status")
async def update_preset_status(
    preset_id: int,
    data: PresetStatusUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")
    preset.status = data.status
    await session.commit()
    await session.refresh(preset)
    return {"id": preset.id, "status": preset.status}
