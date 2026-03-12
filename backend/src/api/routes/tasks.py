from fastapi import APIRouter

from src.api.deps import get_model_manager

router = APIRouter(prefix="/api/v1")


@router.get("/models")
async def list_models():
    manager = get_model_manager()
    return {
        "gpu_status": manager.gpu_status(),
        "models": {
            name: {
                "type": cfg["type"],
                "loaded": manager.is_loaded(name),
                "gpu": cfg["gpu"],
                "vram_gb": cfg["vram_gb"],
            }
            for name, cfg in manager._configs.items()
        },
    }
