import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import load_model_configs, get_settings
from src.gpu.detector import get_device_for_engine, gpu_summary
from src.models.database import get_async_session
from src.models.schemas import EngineInfo, EngineLoadResponse
from src.services import model_scheduler
from src.services.model_metadata_service import (
    get_all_metadata, sync_metadata, refresh_metadata, scan_local_models,
    _format_size,
)
from src.workers.tts_engines.registry import get_engine

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


@router.get("/gpus")
async def list_gpus():
    """Return detected GPU information."""
    return gpu_summary()


def _is_engine_loaded(name: str) -> bool:
    from src.workers.tts_engines import registry as tts_registry
    from src.workers.llm_engines import registry as llm_registry

    engine = tts_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    engine = llm_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    return False


def _build_engine_info(key: str, cfg: dict, meta, local_dirs: set[str]) -> EngineInfo:
    local_path = cfg.get("local_path")
    local_exists = local_path in local_dirs if local_path else False
    info = EngineInfo(
        name=key,
        display_name=cfg["name"],
        type=cfg["type"],
        status="loaded" if _is_engine_loaded(key) else "unloaded",
        gpu=cfg.get("gpu", 1),
        vram_gb=cfg.get("vram_gb", 0),
        resident=cfg.get("resident", False),
        local_path=local_path,
        local_exists=local_exists,
    )
    if meta:
        info.organization = meta.organization
        info.model_size = _format_size(meta.model_size_bytes)
        info.frameworks = meta.frameworks
        info.libraries = meta.libraries
        info.license = meta.license
        info.languages = meta.languages
        info.tags = meta.tags
        info.tensor_types = meta.tensor_types
        info.description = meta.description
        info.has_metadata = True
    return info


@router.get("", response_model=list[EngineInfo])
async def list_all_engines(
    type: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List all engines with metadata. Optionally filter by type."""
    configs = load_model_configs()
    metadata = await get_all_metadata(session)
    local_dirs = scan_local_models()
    result = []
    for key, cfg in configs.items():
        if type and cfg.get("type") != type:
            continue
        result.append(_build_engine_info(key, cfg, metadata.get(key), local_dirs))
    return result


@router.post("/sync-metadata")
async def sync_all_metadata(session: AsyncSession = Depends(get_async_session)):
    """Fetch metadata for any engine not yet in DB."""
    metadata = await sync_metadata(session)
    return {"synced": len(metadata)}


@router.post("/{name}/refresh-metadata", response_model=EngineInfo)
async def refresh_engine_metadata(
    name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Force re-fetch metadata for a specific engine."""
    configs = load_model_configs()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")
    meta = await refresh_metadata(session, name)
    local_dirs = scan_local_models()
    return _build_engine_info(name, configs[name], meta, local_dirs)


@router.post("/{name}/load", response_model=EngineLoadResponse)
async def load_engine(name: str):
    configs = load_model_configs()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    start = time.monotonic()
    await model_scheduler.load_model(name)
    elapsed = round(time.monotonic() - start, 2)

    return EngineLoadResponse(name=name, status="loaded", load_time_seconds=elapsed)


@router.post("/{name}/unload", response_model=EngineLoadResponse)
async def unload_engine(name: str, force: bool = False):
    configs = load_model_configs()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    await model_scheduler.unload_model(name, force=force)

    return EngineLoadResponse(name=name, status="unloaded")


@router.get("/scheduler/status")
async def scheduler_status():
    """Return current model scheduler status."""
    return model_scheduler.get_status()
