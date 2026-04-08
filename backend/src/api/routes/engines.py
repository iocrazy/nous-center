import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.config import get_settings
from src.services.model_scanner import scan_models
from src.gpu.detector import get_device_for_engine, gpu_summary
from src.models.database import get_async_session
from src.models.schemas import EngineInfo, EngineLoadResponse
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


def _get_model_manager(request: Request):
    return getattr(request.app.state, "model_manager", None)


def _is_engine_loaded(name: str, request: Request | None = None) -> bool:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None:
            return mgr.is_loaded(name)
    # Fallback to old registries
    from src.workers.tts_engines import registry as tts_registry
    from src.workers.llm_engines import registry as llm_registry
    engine = tts_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    engine = llm_registry._ENGINE_INSTANCES.get(name)
    if engine is not None:
        return engine.is_loaded
    return False


def _get_loaded_gpu(name: str, request: Request | None = None) -> int | None:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None and mgr.is_loaded(name):
            lm = mgr._models.get(name)
            return lm.gpu_index if lm else None
    return None


def _get_loaded_gpus(name: str, request: Request | None = None) -> list[int] | None:
    if request is not None:
        mgr = _get_model_manager(request)
        if mgr is not None and mgr.is_loaded(name):
            lm = mgr._models.get(name)
            if lm and lm.gpu_indices:
                return lm.gpu_indices
            elif lm:
                return [lm.gpu_index]
    return None


def _build_engine_info(key: str, cfg: dict, meta, local_dirs: set[str], request: Request | None = None) -> EngineInfo:
    local_path = cfg.get("local_path")
    local_exists = local_path in local_dirs if local_path else False
    loaded = _is_engine_loaded(key, request)
    info = EngineInfo(
        name=key,
        display_name=cfg["name"],
        type=cfg["type"],
        status="loaded" if loaded else "unloaded",
        gpu=cfg.get("gpu", 1),
        vram_gb=cfg.get("vram_gb", 0),
        resident=cfg.get("resident", False),
        local_path=local_path,
        local_exists=local_exists,
        auto_detected=cfg.get("auto_detected", False),
        loaded_gpu=_get_loaded_gpu(key, request) if loaded else None,
        loaded_gpus=_get_loaded_gpus(key, request) if loaded else None,
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
    request: Request,
    type: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List all engines with metadata. Optionally filter by type."""
    configs = scan_models()
    metadata = await get_all_metadata(session)
    local_dirs = scan_local_models()
    result = []
    for key, cfg in configs.items():
        if type and cfg.get("type") != type:
            continue
        # Only show models that exist locally
        local_path = cfg.get("local_path")
        if not local_path or local_path not in local_dirs:
            continue
        result.append(_build_engine_info(key, cfg, metadata.get(key), local_dirs, request))
    return result


@router.post("/scan", dependencies=[Depends(require_admin)])
async def scan_models_endpoint():
    """Re-scan models directory for new models."""
    configs = scan_models()
    return {"count": len(configs), "models": list(configs.keys())}


@router.post("/sync-metadata", dependencies=[Depends(require_admin)])
async def sync_all_metadata(session: AsyncSession = Depends(get_async_session)):
    """Fetch metadata for any engine not yet in DB."""
    metadata = await sync_metadata(session)
    return {"synced": len(metadata)}


@router.post("/{name}/refresh-metadata", response_model=EngineInfo, dependencies=[Depends(require_admin)])
async def refresh_engine_metadata(
    name: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Force re-fetch metadata for a specific engine."""
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")
    meta = await refresh_metadata(session, name)
    local_dirs = scan_local_models()
    return _build_engine_info(name, configs[name], meta, local_dirs)


@router.post("/{name}/load", response_model=EngineLoadResponse, dependencies=[Depends(require_admin)])
async def load_engine(name: str, request: Request):
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    model_mgr = request.app.state.model_manager
    start = time.monotonic()
    try:
        await model_mgr.load_model(name)
    except Exception as e:
        raise HTTPException(503, detail=f"加载失败: {e}")
    elapsed = round(time.monotonic() - start, 2)

    return EngineLoadResponse(name=name, status="loaded", load_time_seconds=elapsed)


@router.post("/{name}/unload", response_model=EngineLoadResponse, dependencies=[Depends(require_admin)])
async def unload_engine(name: str, request: Request, force: bool = False):
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    model_mgr = request.app.state.model_manager
    await model_mgr.unload_model(name, force=force)

    return EngineLoadResponse(name=name, status="unloaded")


@router.patch("/{name}/resident", dependencies=[Depends(require_admin)])
async def set_resident(name: str, resident: bool = True):
    """Toggle auto-load on startup for an engine."""
    import yaml

    configs_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"

    with open(configs_path) as f:
        data = yaml.safe_load(f)

    # Support both old dict format and new list format
    models = data.get("models", [])
    if isinstance(models, list):
        found = False
        for entry in models:
            if entry.get("id") == name:
                entry["resident"] = resident
                found = True
                break
        if not found:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
    else:
        if name not in models:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
        models[name]["resident"] = resident

    with open(configs_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    return {"name": name, "resident": resident}


@router.patch("/{name}/gpu", dependencies=[Depends(require_admin)])
async def set_gpu(name: str, gpu: int = 0):
    """Change GPU assignment for an engine."""
    import yaml

    configs_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"

    with open(configs_path) as f:
        data = yaml.safe_load(f)

    # Support both old dict format and new list format
    models = data.get("models", [])
    if isinstance(models, list):
        found = False
        for entry in models:
            if entry.get("id") == name:
                entry["gpu"] = gpu
                found = True
                break
        if not found:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
    else:
        if name not in models:
            raise HTTPException(404, detail=f"Unknown engine: {name}")
        models[name]["gpu"] = gpu

    with open(configs_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    return {"name": name, "gpu": gpu}


@router.get("/scheduler/status")
async def scheduler_status(request: Request):
    """Return current model manager status."""
    model_mgr = request.app.state.model_manager
    return model_mgr.get_status()
