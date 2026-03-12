import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.models.schemas import AudioUploadResponse

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _get_upload_dir() -> Path:
    """Resolve upload directory at call time (testable)."""
    return Path(os.getenv("AUDIO_UPLOAD_DIR", "assets/voices/uploads"))


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, detail=f"Unsupported format: {ext}. Use: {ALLOWED_EXTENSIONS}")

    file_id = str(uuid.uuid4())
    upload_dir = _get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{file_id}{ext}"

    content = await file.read()
    dest.write_bytes(content)

    return AudioUploadResponse(id=file_id, path=str(dest))


@router.get("/{audio_id}")
async def get_audio_info(audio_id: str):
    upload_dir = _get_upload_dir()
    matches = list(upload_dir.glob(f"{audio_id}.*"))
    if not matches:
        raise HTTPException(404, detail=f"Audio not found: {audio_id}")
    path = matches[0]
    return {"id": audio_id, "path": str(path), "filename": path.name}
