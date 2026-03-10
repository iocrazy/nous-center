import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import (
    ImageGenerateRequest,
    VideoGenerateRequest,
    TTSRequest,
)

router = APIRouter(prefix="/api/v1/generate")


def dispatch_task(task_type: str, params: dict) -> str:
    """Dispatch a task to Celery. Returns task_id."""
    task_id = str(uuid.uuid4())
    # Celery integration will be wired in Task 12
    return task_id


@router.post("/image", status_code=202)
async def generate_image(req: ImageGenerateRequest):
    task_id = dispatch_task("image", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "image"},
    )


@router.post("/video", status_code=202)
async def generate_video(req: VideoGenerateRequest):
    task_id = dispatch_task("video", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "video"},
    )


@router.post("/tts", status_code=202)
async def generate_tts(req: TTSRequest):
    task_id = dispatch_task("tts", req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "tts"},
    )
