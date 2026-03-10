import base64
import time
import uuid
import uuid as uuid_mod

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import (
    BatchTaskInfo,
    BatchTTSRequest,
    BatchTTSResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TTSRequest,
)
from src.models.voice_preset import VoicePreset
from src.workers.tts_worker import generate_tts_task

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


@router.post("/generate", status_code=202)
async def tts_generate(req: TTSRequest):
    """Dispatch async TTS generation task via Celery."""
    task_id = str(uuid.uuid4())
    generate_tts_task.delay(task_id, req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "tts"},
    )


def _get_loaded_engine(name: str):
    """Get a loaded engine or None."""
    from src.workers.tts_engines import registry

    engine = registry._ENGINE_INSTANCES.get(name)
    if engine and engine.is_loaded:
        return engine
    return None


@router.post("/synthesize", response_model=SynthesizeResponse)
async def tts_synthesize(req: SynthesizeRequest):
    """Synchronous TTS synthesis for debugging. Returns audio as base64."""
    engine = _get_loaded_engine(req.engine)
    if engine is None:
        raise HTTPException(
            409,
            detail=f"Engine {req.engine} not loaded. POST /api/v1/engines/{req.engine}/load first.",
        )

    start = time.monotonic()
    result = engine.synthesize(
        text=req.text,
        voice=req.voice,
        speed=req.speed,
        sample_rate=req.sample_rate,
        reference_audio=req.reference_audio,
    )
    elapsed = time.monotonic() - start
    rtf = round(elapsed / max(result.duration_seconds, 0.01), 4) or 0.01

    return SynthesizeResponse(
        audio_base64=base64.b64encode(result.audio_bytes).decode(),
        sample_rate=result.sample_rate,
        duration_seconds=result.duration_seconds,
        engine=req.engine,
        rtf=rtf,
        format=result.format,
    )


async def _resolve_preset(name: str, session: AsyncSession) -> dict | None:
    """Look up voice preset by name. Returns dict with engine + params or None."""
    result = await session.execute(
        select(VoicePreset).where(VoicePreset.name == name)
    )
    preset = result.scalar_one_or_none()
    if preset is None:
        return None
    return {"engine": preset.engine, "params": preset.params or {}}


@router.post("/batch", response_model=BatchTTSResponse, status_code=202)
async def tts_batch(
    req: BatchTTSRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Dispatch multiple TTS tasks for multi-character scenarios."""
    batch_id = f"batch_{uuid_mod.uuid4().hex[:12]}"
    tasks = []

    for i, segment in enumerate(req.segments):
        preset = await _resolve_preset(segment.voice_preset, session)
        if preset is None:
            raise HTTPException(
                404, detail=f"Voice preset not found: {segment.voice_preset}"
            )

        params = {
            **preset["params"],
            "text": segment.text,
            "engine": preset["engine"],
        }
        task = generate_tts_task.delay(f"{batch_id}_{i}", params)
        tasks.append(BatchTaskInfo(index=i, task_id=task.id))

    return BatchTTSResponse(batch_id=batch_id, tasks=tasks)
