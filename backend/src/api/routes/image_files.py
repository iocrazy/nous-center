"""GET /files/images/{date}/{uuid}.{ext}?token=&expires= — signed-URL
static route for image_generate outputs.

Sits OUTSIDE the /api/v1/* tree so admin auth doesn't apply (the URL
itself is the auth via HMAC). The token binds (uuid, expires) so a
leaked URL can't be modified to point at a different file or extend
its lifetime.
"""
from __future__ import annotations

import re

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.errors import NotFoundError, NousError
from src.services.image_output_storage import resolve_path, verify_token

router = APIRouter(tags=["image-files"])


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UUID_RE = re.compile(r"^[0-9a-f]{32}$")
_EXT_WHITELIST = {"png", "jpg", "jpeg", "webp"}


class _UrlExpiredError(NousError):
    type = "url_expired"
    http_status = 403


class _UrlInvalidError(NousError):
    type = "invalid_request_error"
    http_status = 403


_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


@router.get("/files/images/{date}/{uuid_filename}")
async def get_image(date: str, uuid_filename: str, token: str = "", expires: int = 0):
    # Validate URL components BEFORE touching disk so a malformed request
    # can't be used to probe the filesystem layout.
    if not _DATE_RE.match(date):
        raise _UrlInvalidError("invalid date segment", code="bad_url")

    stem, sep, ext = uuid_filename.rpartition(".")
    if not sep or not stem or ext.lower() not in _EXT_WHITELIST:
        raise _UrlInvalidError("invalid filename", code="bad_url")
    ext = ext.lower()
    if not _UUID_RE.match(stem):
        raise _UrlInvalidError("invalid uuid", code="bad_url")

    if not token or not expires:
        raise _UrlInvalidError(
            "signed url requires token and expires query params",
            code="missing_signature",
            fix="Re-fetch the image URL from the workflow output — tokens expire after 1h",
        )

    if not verify_token(stem, expires, token):
        # Same error class for tampered + expired so we don't leak which
        # case it was. The fix hint covers both.
        raise _UrlExpiredError(
            "signed url is expired or tampered",
            code="url_expired",
            fix="Re-run the workflow to get a fresh URL (default TTL 1h)",
        )

    path = resolve_path(date, stem, ext)
    if not path.exists():
        raise NotFoundError("image not found", code="image_missing")

    return FileResponse(
        path=str(path),
        media_type=_MIME_BY_EXT[ext],
        # Aggressive cache OK: the URL changes when expires changes, so a
        # cached response can never outlive its own signature window.
        headers={"Cache-Control": f"private, max-age={max(0, expires - _now_ts())}"},
    )


def _now_ts() -> int:
    import time
    return int(time.time())
