"""External preset service endpoints (Bearer Token auth required)."""

import base64
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_preset_key
from src.models.database import get_async_session
from src.models.preset_api_key import PresetApiKey
from src.models.voice_preset import VoicePreset

router = APIRouter(prefix="/v1/preset", tags=["preset-service"])


class PresetSynthesizeRequest(BaseModel):
    text: str
    emotion: str | None = None


@router.post("/{preset_id}/synthesize")
async def preset_synthesize(
    req: PresetSynthesizeRequest,
    auth: tuple[VoicePreset, PresetApiKey] = Depends(verify_preset_key),
    session: AsyncSession = Depends(get_async_session),
):
    preset, api_key = auth

    # Resolve engine
    from src.workers.tts_engines import registry

    engine_name = preset.engine
    engine = registry._ENGINE_INSTANCES.get(engine_name)
    if not engine or not engine.is_loaded:
        raise HTTPException(
            409,
            detail=f"Engine {engine_name} not loaded. Load it via the management API first.",
        )

    # Build synthesis kwargs from preset params
    params = preset.params or {}
    kwargs = {
        "text": req.text,
        "voice": params.get("voice", "default"),
        "speed": params.get("speed", 1.0),
        "sample_rate": params.get("sample_rate", 24000),
    }
    if preset.reference_audio_path:
        kwargs["reference_audio"] = preset.reference_audio_path
    if preset.reference_text:
        kwargs["reference_text"] = preset.reference_text
    if req.emotion is not None:
        kwargs["emotion"] = req.emotion

    # Synthesize
    start = time.monotonic()
    result = engine.synthesize(**kwargs)
    elapsed = time.monotonic() - start
    rtf = round(elapsed / max(result.duration_seconds, 0.01), 4)

    # Update usage counters
    api_key.usage_calls += 1
    api_key.usage_chars += len(req.text)
    api_key.last_used_at = datetime.now(timezone.utc)
    await session.commit()

    audio_b64 = base64.b64encode(result.audio_bytes).decode()
    return {
        "audio_base64": audio_b64,
        "sample_rate": result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "engine": engine_name,
        "rtf": rtf,
        "format": result.format,
    }
