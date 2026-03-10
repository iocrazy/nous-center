import base64
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.models.schemas import SynthesizeRequest, SynthesizeResponse, TTSRequest
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
