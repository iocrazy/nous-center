"""GET /files/images/{date}/{uuid}.{ext}?token=&expires= — signed-URL
static route for image_generate outputs.

Sits OUTSIDE the /api/v1/* tree so admin gate middleware doesn't apply.
Authentication is dual-path:

  1. **Admin session cookie** — owner's own browser. Bypasses TTL so the
     UI doesn't break when a workflow output sits open for >1h then the
     React Query cache is re-mounted with a now-expired token URL.
  2. **HMAC signed URL** — anonymous / external share path. The token
     binds (uuid, expires) so a leaked URL can't be modified to point at
     a different file or extend its lifetime.

Either path is sufficient. Ordering: cookie first (cheaper, no signature
work), HMAC second (still required for non-admin URLs).
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from src.api.admin_session import request_is_authed
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
async def get_image(
    request: Request,
    date: str,
    uuid_filename: str,
    token: str = "",
    expires: int = 0,
):
    # Validate URL components BEFORE touching disk so a malformed request
    # can't be used to probe the filesystem layout. These checks run
    # regardless of auth path so a cookie-authed admin still can't smuggle
    # ../ paths through.
    if not _DATE_RE.match(date):
        raise _UrlInvalidError("invalid date segment", code="bad_url")

    stem, sep, ext = uuid_filename.rpartition(".")
    if not sep or not stem or ext.lower() not in _EXT_WHITELIST:
        raise _UrlInvalidError("invalid filename", code="bad_url")
    ext = ext.lower()
    if not _UUID_RE.match(stem):
        raise _UrlInvalidError("invalid uuid", code="bad_url")

    # Dual auth: admin session cookie OR HMAC signed URL.
    # Cookie path is checked first (cheap; no HMAC math). When ADMIN_PASSWORD is
    # empty (dev mode), request_is_authed returns True so dev keeps working
    # without configuring crypto.
    admin_authed = request_is_authed(request)
    if not admin_authed:
        if not token or not expires:
            raise _UrlInvalidError(
                "image url requires admin session or signed token+expires",
                code="missing_signature",
                fix="Log in as admin, or re-fetch the workflow output for a fresh signed URL",
            )
        if not verify_token(stem, expires, token):
            # Same error class for tampered + expired so we don't leak which
            # case it was. The fix hint covers both.
            raise _UrlExpiredError(
                "signed url is expired or tampered",
                code="url_expired",
                fix="Log in as admin, or re-run the workflow for a fresh URL (default TTL 1h)",
            )

    path = resolve_path(date, stem, ext)
    if not path.exists():
        raise NotFoundError("image not found", code="image_missing")

    # Cache policy:
    # - HMAC path: aggressive cache OK; URL changes when expires changes so a
    #   cached response can never outlive its own signature window.
    # - Cookie path: short cache (60s) — same uuid URL serves repeatedly across
    #   sessions, can't anchor cache to a signature window.
    if admin_authed and not (token and expires and verify_token(stem, expires, token)):
        cache_header = "private, max-age=60"
    else:
        cache_header = f"private, max-age={max(0, expires - _now_ts())}"

    return FileResponse(
        path=str(path),
        media_type=_MIME_BY_EXT[ext],
        headers={"Cache-Control": cache_header},
    )


def _now_ts() -> int:
    import time
    return int(time.time())
