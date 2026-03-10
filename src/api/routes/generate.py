import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import (
    ImageGenerateRequest,
    VideoGenerateRequest,
)
from src.workers.image_worker import generate_image_task
from src.workers.video_worker import generate_video_task

router = APIRouter(prefix="/api/v1/generate")

TASK_MAP = {
    "image": generate_image_task,
    "video": generate_video_task,
}


def dispatch_task(task_type: str, params: dict) -> str:
    """Dispatch a task to Celery. Returns task_id."""
    task_id = str(uuid.uuid4())
    celery_task = TASK_MAP[task_type]
    celery_task.delay(task_id, params)
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


