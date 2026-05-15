"""External instance service endpoints (Bearer Token auth required)."""

import base64
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
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


@router.post("/{instance_id}/run", status_code=202)
async def instance_run(
    req: InstanceRunRequest,
    request: Request,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_instance_key),
    session: AsyncSession = Depends(get_async_session),
):
    """入队一个已发布 workflow 实例的执行（D17 纯异步）。

    返回 202 + task_id。客户端轮询 GET /api/v1/tasks/{task_id} 或订阅
    WS /ws/workflow/{instance_id} 拿结果。迁移指引见 docs/run-async-migration.md。
    """
    import asyncio

    from src.models.execution_task import ExecutionTask
    from src.services.workflow_runner import run_workflow_task

    instance, api_key = auth

    if instance.source_type != "workflow":
        raise HTTPException(
            400, detail="Only workflow-based instances support /run"
        )

    workflow_data = instance.params_override or {}
    nodes = workflow_data.get("nodes", [])
    if not nodes:
        raise HTTPException(400, detail="Workflow has no nodes")

    task = ExecutionTask(
        workflow_id=instance.source_id,
        workflow_name=instance.name or "API 执行",
        status="queued",
        nodes_total=len(nodes),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    # 用量计数：入队即计一次调用（异步契约下无法等执行完）
    api_key.usage_calls += 1
    api_key.last_used_at = datetime.now(timezone.utc)
    await session.commit()

    # WS channel 仍用 instance.id（上游订阅 /ws/workflow/{instance_id} 的老约定不变），
    # 但同时把 task_id 回给客户端用于轮询。
    channel_id = str(instance.id)
    runner_client = getattr(request.app.state, "runner_client", None)

    asyncio.create_task(run_workflow_task(
        task.id, workflow_data,
        runner_client=runner_client, channel_id=channel_id,
    ))
    return {"task_id": str(task.id), "status": "queued"}
