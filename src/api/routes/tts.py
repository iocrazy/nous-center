import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.schemas import TTSRequest
from src.workers.tts_worker import generate_tts_task

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


@router.post("/generate", status_code=202)
async def tts_generate(req: TTSRequest):
    """Dispatch async TTS generation task via Celery."""
    task_id = str(uuid.uuid4())
    generate_tts_task.delay(task_id, req.model_dump())
    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "status": "pending", "type": "tts"},
    )
