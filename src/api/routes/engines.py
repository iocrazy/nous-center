import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.config import load_model_configs, get_settings
from src.gpu.detector import get_device_for_engine, gpu_summary
from src.models.schemas import EngineInfo, EngineLoadResponse
from src.workers.tts_engines.registry import get_engine

router = APIRouter(prefix="/api/v1/engines", tags=["engines"])


@router.get("/gpus")
async def list_gpus():
    """Return detected GPU information."""
    return gpu_summary()


def _is_engine_loaded(name: str) -> bool:
    from src.workers.tts_engines import registry
    engine = registry._ENGINE_INSTANCES.get(name)
    return engine is not None and engine.is_loaded


@router.get("", response_model=list[EngineInfo])
async def list_all_engines():
    configs = load_model_configs()
    result = []
    for key, cfg in configs.items():
        if cfg.get("type") != "tts":
            continue
        result.append(EngineInfo(
            name=key,
            display_name=cfg["name"],
            type=cfg["type"],
            status="loaded" if _is_engine_loaded(key) else "unloaded",
            gpu=cfg.get("gpu", 1),
            vram_gb=cfg.get("vram_gb", 0),
            resident=cfg.get("resident", False),
        ))
    return result


@router.post("/{name}/load", response_model=EngineLoadResponse)
async def load_engine(name: str):
    configs = load_model_configs()
    if name not in configs or configs[name].get("type") != "tts":
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    settings = get_settings()
    model_path = Path(settings.LOCAL_MODELS_PATH) / cfg["local_path"]
    device = get_device_for_engine(cfg)

    start = time.monotonic()
    engine = get_engine(name, model_path=model_path, device=device)
    if not engine.is_loaded:
        await asyncio.to_thread(engine.load)
    elapsed = round(time.monotonic() - start, 2)

    return EngineLoadResponse(name=name, status="loaded", load_time_seconds=elapsed)


@router.post("/{name}/unload", response_model=EngineLoadResponse)
async def unload_engine(name: str, force: bool = False):
    configs = load_model_configs()
    if name not in configs or configs[name].get("type") != "tts":
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    from src.workers.tts_engines import registry
    engine = registry._ENGINE_INSTANCES.get(name)
    if engine and engine.is_loaded:
        engine.unload()

    return EngineLoadResponse(name=name, status="unloaded")
