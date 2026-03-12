"""Bearer Token authentication for preset service endpoints."""

import bcrypt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.preset_api_key import PresetApiKey
from src.models.voice_preset import VoicePreset


async def verify_preset_key(
    preset_id: int,
    authorization: str = Header(...),
    session: AsyncSession = Depends(get_async_session),
) -> tuple[VoicePreset, PresetApiKey]:
    """Verify Bearer token against preset API keys.

    Returns (preset, matched_key) on success.
    Raises 401/403/404 on failure.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Invalid authorization header")
    token = authorization[7:]

    preset = await session.get(VoicePreset, preset_id)
    if not preset:
        raise HTTPException(404, detail="Preset not found")
    if preset.status != "active":
        raise HTTPException(403, detail="Preset is inactive")

    result = await session.execute(
        select(PresetApiKey).where(
            PresetApiKey.preset_id == preset_id,
            PresetApiKey.is_active == True,  # noqa: E712
        )
    )
    keys = result.scalars().all()

    for key in keys:
        if bcrypt.checkpw(token.encode(), key.key_hash.encode()):
            return preset, key

    raise HTTPException(401, detail="Invalid API key")
