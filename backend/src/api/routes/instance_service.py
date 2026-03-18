"""External instance service endpoints (Bearer Token auth required)."""

import base64
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_instance_key
from src.models.database import get_async_session
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance
from src.models.voice_preset import VoicePreset

router = APIRouter(prefix="/v1/instances", tags=["instance-service"])


class InstanceSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    emotion: str | None = None


@router.post("/{instance_id}/synthesize")
async def instance_synthesize(
    req: InstanceSynthesizeRequest,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_instance_key),
    session: AsyncSession = Depends(get_async_session),
):
    instance, api_key = auth

    # Load preset to get engine config (resolve from source_type)
    if instance.source_type != "preset":
        raise HTTPException(501, detail="Only preset-based instances support synthesis currently")
    preset = await session.get(VoicePreset, instance.source_id)
    if not preset:
        raise HTTPException(500, detail="Linked preset not found")

    # Resolve engine
    from src.workers.tts_engines import registry

    engine_name = preset.engine
    engine = registry._ENGINE_INSTANCES.get(engine_name)
    if not engine or not engine.is_loaded:
        raise HTTPException(
            409,
            detail=f"Engine {engine_name} not loaded. Load it via the management API first.",
        )

    # Merge preset params with instance overrides
    params = {**(preset.params or {}), **(instance.params_override or {})}
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


class InstanceRunRequest(BaseModel):
    inputs: dict | None = None


@router.post("/{instance_id}/run")
async def instance_run(
    req: InstanceRunRequest,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_instance_key),
    session: AsyncSession = Depends(get_async_session),
):
    """Execute a published workflow instance."""
    from src.services.workflow_executor import WorkflowExecutor, ExecutionError

    instance, api_key = auth

    if instance.source_type != "workflow":
        raise HTTPException(
            400, detail="Only workflow-based instances support /run"
        )

    # The workflow DAG is stored in params_override at publish time
    workflow_data = instance.params_override or {}
    if not workflow_data.get("nodes"):
        raise HTTPException(400, detail="Workflow has no nodes")

    async def broadcast_progress(data: dict):
        from src.api.main import _ws_connections
        for ws in _ws_connections.get(str(instance.id), []):
            try:
                await ws.send_json(data)
            except Exception:
                pass

    executor = WorkflowExecutor(workflow_data, on_progress=broadcast_progress)

    try:
        result = await executor.execute()
    except ExecutionError as e:
        raise HTTPException(422, detail=str(e))

    await broadcast_progress({"type": "complete", "progress": 100})

    # Update usage counters
    api_key.usage_calls += 1
    api_key.last_used_at = datetime.now(timezone.utc)
    await session.commit()

    return result
