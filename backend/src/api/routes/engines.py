import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_admin import require_admin
from src.services.model_scanner import scan_models
from src.gpu.detector import gpu_summary
from src.models.database import get_async_session
from src.models.schemas import EngineInfo, EngineLoadResponse
from src.services.model_metadata_service import (
    get_all_metadata, sync_metadata, refresh_metadata, scan_local_models,
    _format_size,
)
from src.api.websocket import ws_manager

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])

# In-memory loading state tracker: model_id -> {"status": "loading"|"failed", "detail": str}
_loading_states: dict[str, dict[str, str]] = {}


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

    # Determine status: check loading states first, then fall back to loaded/unloaded
    loading_state = _loading_states.get(key)
    if loading_state:
        status = loading_state["status"]
        status_detail = loading_state.get("detail", "")
    elif loaded:
        status = "loaded"
        status_detail = None
    else:
        status = "unloaded"
        status_detail = None

    info = EngineInfo(
        name=key,
        display_name=cfg["name"],
        type=cfg["type"],
        status=status,
        gpu=cfg.get("gpu", 1),
        vram_gb=cfg.get("vram_gb", 0),
        resident=cfg.get("resident", False),
        local_path=local_path,
        local_exists=local_exists,
        auto_detected=cfg.get("auto_detected", False),
        loaded_gpu=_get_loaded_gpu(key, request) if loaded else None,
        loaded_gpus=_get_loaded_gpus(key, request) if loaded else None,
        status_detail=status_detail,
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


@router.post("/reload", dependencies=[Depends(require_admin)])
async def reload_registry(request: Request):
    """Hot-reload models.yaml without restarting. Picks up new model configs."""
    mgr = _get_model_manager(request)
    if mgr is None:
        raise HTTPException(503, "ModelManager not initialized")
    new_count = mgr._registry.reload()
    return {"status": "reloaded", "new_models": new_count, "total": len(mgr._registry.specs)}


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

    # Reject if already loading
    if name in _loading_states and _loading_states[name]["status"] == "loading":
        return EngineLoadResponse(name=name, status="loading")

    model_mgr = request.app.state.model_manager

    # If already loaded, return immediately
    if model_mgr.is_loaded(name):
        return EngineLoadResponse(name=name, status="loaded")

    # Start background loading
    _loading_states[name] = {"status": "loading", "detail": "Starting..."}
    asyncio.create_task(_load_in_background(name, model_mgr))

    return EngineLoadResponse(name=name, status="loading")


async def _load_in_background(name: str, model_mgr):
    import logging
    logger = logging.getLogger(__name__)
    start = time.monotonic()
    try:
        _loading_states[name] = {"status": "loading", "detail": "Loading model..."}
        await ws_manager.broadcast_model_status(name, "loading", "Loading model...")
        await model_mgr.load_model(name)
        elapsed = round(time.monotonic() - start, 2)
        # Clear loading state on success — the model is now truly loaded
        _loading_states.pop(name, None)
        await ws_manager.broadcast_model_status(name, "loaded", f"Ready ({elapsed}s)")
        logger.info("Model %s loaded in %.2fs", name, elapsed)
    except Exception as e:
        _loading_states[name] = {"status": "failed", "detail": str(e)}
        await ws_manager.broadcast_model_status(name, "failed", str(e))
        logger.error("Model %s load failed: %s", name, e)


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


_install_states: dict[str, dict[str, str]] = {}  # engine -> {status, detail}


@router.get("/deps")
async def list_engine_deps():
    """Return install/probe status for every TTS engine in the manifest."""
    from src.services.tts_deps import list_manifest
    data = list_manifest()
    # overlay any in-flight install state
    for k, v in _install_states.items():
        if k in data:
            data[k]["install_state"] = v
    return data


@router.post("/{name}/install_deps", dependencies=[Depends(require_admin)])
async def install_engine_deps(name: str):
    """Install pip deps for a TTS engine (background). Status pushed via ws."""
    from src.services.tts_deps import get as get_dep
    if get_dep(name) is None:
        raise HTTPException(404, detail=f"No dep manifest for engine: {name}")
    state = _install_states.get(name)
    if state and state.get("status") == "installing":
        return {"name": name, "status": "installing"}
    _install_states[name] = {"status": "installing", "detail": "Starting..."}
    asyncio.create_task(_install_in_background(name))
    return {"name": name, "status": "installing"}


async def _install_in_background(name: str):
    import logging as _lg
    from src.services.tts_deps import install
    log = _lg.getLogger(__name__)
    await ws_manager.broadcast_model_status(name, "installing", "Installing deps...")

    async def _push(line: str):
        # Throttle: only push lines that look meaningful (avoid noise)
        if any(k in line.lower() for k in ("collecting", "downloading", "installing", "successfully", "error")):
            _install_states[name] = {"status": "installing", "detail": line[:200]}
            await ws_manager.broadcast_model_status(name, "installing", line[:200])

    try:
        ok, output = await install(name, on_log=_push)
        if ok:
            _install_states[name] = {"status": "installed", "detail": "Install complete"}
            await ws_manager.broadcast_model_status(name, "installed", "Install complete")
            log.info("TTS deps installed for %s", name)
        else:
            tail = "\n".join(output.splitlines()[-5:])
            _install_states[name] = {"status": "install_failed", "detail": tail}
            await ws_manager.broadcast_model_status(name, "install_failed", tail)
            log.error("TTS dep install failed for %s: %s", name, tail)
    except Exception as e:
        _install_states[name] = {"status": "install_failed", "detail": str(e)}
        await ws_manager.broadcast_model_status(name, "install_failed", str(e))
        log.exception("TTS dep install crashed for %s", name)


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
