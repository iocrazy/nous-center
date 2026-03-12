import base64
import json as json_module
import time
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import (
    BatchStatusResponse,
    BatchTaskInfo,
    BatchTTSRequest,
    BatchTTSResponse,
    StreamRequest,
    SynthesizeRequest,
    SynthesizeResponse,
    TTSRequest,
)
from src.models.voice_preset import VoicePreset
from src.workers.tts_worker import generate_tts_task
import redis.asyncio as aioredis
from src.services.tts_cache import make_cache_key, TTSCacheService

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


# Module-level lazy Redis + cache
_redis_client = None
_cache_service = None

def _get_cache_service() -> TTSCacheService | None:
    global _redis_client, _cache_service
    if _cache_service is not None:
        return _cache_service
    try:
        from src.config import get_settings
        _redis_client = aioredis.from_url(get_settings().REDIS_URL)
        _cache_service = TTSCacheService(_redis_client)
        return _cache_service
    except Exception:
        return None


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

    # --- Cache check ---
    cache_key = None
    cache_svc = _get_cache_service() if req.cache else None
    if cache_svc:
        cache_key = make_cache_key(
            text=req.text, engine=req.engine, voice=req.voice,
            speed=req.speed, sample_rate=req.sample_rate, emotion=req.emotion,
        )
        cached = await cache_svc.get(cache_key)
        if cached:
            return SynthesizeResponse(
                audio_base64=cached, sample_rate=req.sample_rate,
                duration_seconds=0, engine=req.engine, rtf=0, cached=True,
            )

    # --- Synthesize ---
    start = time.monotonic()
    kwargs = dict(text=req.text, voice=req.voice, speed=req.speed,
                  sample_rate=req.sample_rate, reference_audio=req.reference_audio)
    if req.reference_text is not None:
        kwargs["reference_text"] = req.reference_text
    if req.emotion is not None:
        kwargs["emotion"] = req.emotion

    result = engine.synthesize(**kwargs)
    elapsed = time.monotonic() - start
    rtf = round(elapsed / max(result.duration_seconds, 0.01), 4)
    audio_b64 = base64.b64encode(result.audio_bytes).decode()

    # --- Cache store ---
    if cache_svc and cache_key:
        try:
            await cache_svc.set(cache_key, audio_b64)
        except Exception:
            pass  # cache write failure is non-fatal

    # --- Usage recording (fire-and-forget) ---
    # Usage is recorded in Task 8's usage_recorder via background task
    # (actual wiring added after Task 8)

    return SynthesizeResponse(
        audio_base64=audio_b64, sample_rate=result.sample_rate,
        duration_seconds=result.duration_seconds, engine=req.engine,
        rtf=rtf, format=result.format, cached=False,
    )


@router.post("/stream")
async def tts_stream(req: StreamRequest):
    """SSE streaming TTS synthesis.

    Note: Currently single-chunk (engine.synthesize returns complete audio).
    The SSE format enables future true streaming when engines support
    synthesize_stream() yielding multiple chunks.
    """
    engine = _get_loaded_engine(req.engine)
    if engine is None:
        raise HTTPException(
            409,
            detail=f"Engine {req.engine} not loaded. POST /api/v1/engines/{req.engine}/load first.",
        )

    async def event_generator():
        try:
            start = time.monotonic()
            kwargs = dict(
                text=req.text,
                voice=req.voice,
                speed=req.speed,
                sample_rate=req.sample_rate,
                reference_audio=req.reference_audio,
            )
            if req.reference_text is not None:
                kwargs["reference_text"] = req.reference_text
            if req.emotion is not None:
                kwargs["emotion"] = req.emotion

            result = engine.synthesize(**kwargs)
            elapsed = time.monotonic() - start
            rtf = round(elapsed / max(result.duration_seconds, 0.01), 4)

            audio_b64 = base64.b64encode(result.audio_bytes).decode()
            chunk = json_module.dumps({"seq": 1, "audio": audio_b64, "format": result.format})
            yield f"event: audio\ndata: {chunk}\n\n"

            done = json_module.dumps({
                "total_chunks": 1,
                "duration_ms": int(result.duration_seconds * 1000),
                "usage": {"characters": len(req.text), "rtf": rtf},
            })
            yield f"event: done\ndata: {done}\n\n"
        except Exception as exc:
            error = json_module.dumps({"code": "ENGINE_ERROR", "message": str(exc)})
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
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


@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def tts_batch_status(batch_id: str):
    """Query status of a batch TTS job by checking Celery task results."""
    from src.workers.celery_app import celery_app

    # Discover task IDs by probing batch_id_{0..N} pattern
    tasks = []
    for i in range(100):  # reasonable upper bound
        task_id = f"{batch_id}_{i}"
        result = AsyncResult(task_id, app=celery_app)
        if result.state == "PENDING" and result.result is None and i > 0:
            # No more tasks in this batch
            break
        tasks.append({
            "index": i,
            "task_id": task_id,
            "status": result.state.lower(),
            "result": result.result if result.ready() else None,
        })

    if not tasks:
        raise HTTPException(404, detail=f"Batch not found: {batch_id}")

    statuses = {t["status"] for t in tasks}
    if all(s == "success" for s in statuses):
        overall = "completed"
    elif "failure" in statuses:
        overall = "failed"
    elif "success" in statuses:
        overall = "partial"
    else:
        overall = "pending"

    return BatchStatusResponse(batch_id=batch_id, status=overall, tasks=tasks)
