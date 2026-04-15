"""Files API (Step 5) — upload, get, list, download, delete.

Content-addressed local storage at `backend/data/files/{sha256[:2]}/{sha256}`.
Dedup per (instance_id, sha256): repeated uploads return the same `file-xxx` id.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File as UploadField, Form, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps_auth import verify_bearer_token
from src.errors import (
    InvalidRequestError, NotFoundError,
    PermissionError as NousPermissionError,
    NousError,
)
from src.models.database import get_async_session
from src.models.file import File as FileRow
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50MB
CHUNK = 1024 * 1024  # 1MB

ALLOWED_PURPOSES = {"user_data", "assistants", "batch", "vision"}

# Repo root: backend/src/api/routes/files.py -> up 4 = backend/
_BACKEND_DIR = Path(__file__).resolve().parents[3]
FILES_ROOT = _BACKEND_DIR / "data" / "files"


def _new_file_id() -> str:
    return f"file-{secrets.token_urlsafe(12)}"


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _render(row: FileRow) -> dict:
    return {
        "id": row.id,
        "object": "file",
        "bytes": row.bytes,
        "created_at": int(_to_utc(row.created_at).timestamp()),
        "filename": row.filename,
        "purpose": row.purpose,
    }


def _storage_path_for(sha256: str) -> Path:
    return FILES_ROOT / sha256[:2] / sha256


class _PayloadTooLarge(NousError):
    type = "invalid_request_error"
    http_status = 413


@router.post("/v1/files")
async def upload_file(
    file: UploadFile = UploadField(...),
    purpose: str = Form(...),
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    if purpose not in ALLOWED_PURPOSES:
        raise InvalidRequestError(
            f"purpose must be one of {sorted(ALLOWED_PURPOSES)}",
            code="invalid_purpose",
            param="purpose",
        )

    FILES_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = FILES_ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}"

    hasher = hashlib.sha256()
    total = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise _PayloadTooLarge(
                        f"file exceeds {MAX_FILE_BYTES} bytes",
                        code="file_too_large",
                    )
                hasher.update(chunk)
                out.write(chunk)
        sha = hasher.hexdigest()

        # Dedup check
        existing = (await session.execute(
            select(FileRow).where(
                FileRow.instance_id == instance.id,
                FileRow.sha256 == sha,
            )
        )).scalar_one_or_none()
        if existing is not None:
            return _render(existing)

        # Move tmp -> content-addressed path (shared across instances OK;
        # other instance may have already placed the same content)
        dest = _storage_path_for(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.replace(tmp_path, dest)
        else:
            tmp_path.unlink(missing_ok=True)

        row = FileRow(
            id=_new_file_id(),
            instance_id=instance.id,
            purpose=purpose,
            filename=file.filename or "unnamed",
            bytes=total,
            mime_type=file.content_type or "application/octet-stream",
            sha256=sha,
            storage_path=str(dest),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return _render(row)
    except _PayloadTooLarge:
        raise
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


async def _load_owned(session: AsyncSession, file_id: str, instance_id: int) -> FileRow:
    row = await session.get(FileRow, file_id)
    if row is None:
        raise NotFoundError("file not found", code="file_not_found")
    if row.instance_id != instance_id:
        raise NousPermissionError(
            "file belongs to another instance",
            code="file_wrong_instance",
        )
    return row


@router.get("/v1/files/{file_id}")
async def get_file(
    file_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    row = await _load_owned(session, file_id, instance.id)
    return _render(row)


@router.get("/v1/files/{file_id}/content")
async def download_file(
    file_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    row = await _load_owned(session, file_id, instance.id)
    p = Path(row.storage_path)
    if not p.exists():
        raise NotFoundError(
            "file content missing on disk", code="file_content_missing"
        )
    return FileResponse(
        path=str(p),
        media_type=row.mime_type,
        filename=row.filename,
    )


@router.get("/v1/files")
async def list_files(
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
    purpose: str | None = None,
    limit: int = 20,
    after: str | None = None,
):
    instance, _ = auth
    limit = max(1, min(limit, 100))
    stmt = select(FileRow).where(FileRow.instance_id == instance.id)
    if purpose:
        stmt = stmt.where(FileRow.purpose == purpose)
    if after:
        anchor = await session.get(FileRow, after)
        if anchor is None or anchor.instance_id != instance.id:
            raise InvalidRequestError(
                "after cursor not found",
                code="invalid_cursor",
                param="after",
            )
        stmt = stmt.where(
            tuple_(FileRow.created_at, FileRow.id)
            < (anchor.created_at, anchor.id)
        )
    stmt = stmt.order_by(
        FileRow.created_at.desc(), FileRow.id.desc()
    ).limit(limit + 1)
    rows = (await session.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    data = [_render(r) for r in rows]
    return {
        "object": "list",
        "data": data,
        "has_more": has_more,
        "last_id": data[-1]["id"] if data else None,
    }


@router.delete("/v1/files/{file_id}")
async def delete_file(
    file_id: str,
    auth: tuple[ServiceInstance, InstanceApiKey] = Depends(verify_bearer_token),
    session: AsyncSession = Depends(get_async_session),
):
    instance, _ = auth
    row = await _load_owned(session, file_id, instance.id)
    # Don't touch disk blob: other instances may share the same sha256.
    # Orphan reaper (future) can GC unreferenced blobs.
    await session.delete(row)
    await session.commit()
    return {"id": file_id, "object": "file.deleted", "deleted": True}
