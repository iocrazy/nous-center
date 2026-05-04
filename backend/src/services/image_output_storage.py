"""Image output storage with signed URLs.

write_image(png_bytes, ext='png') →
    {uuid, path, url}        when ADMIN_SESSION_SECRET is configured
    {uuid, path, url=None}   in dev mode (no secret) — caller should fall
                             back to base64 inline so tests / fresh installs
                             still work end-to-end.

URL shape:
    /files/images/{YYYY-MM-DD}/{uuid}.{ext}?token={hmac}&expires={unix_ts}

The HMAC binds (uuid, expires) so a token never grants access to a
different file or a different time window. Default TTL is 1h; admin can
override per-image via the optional `ttl_seconds` arg (DiffusersImageBackend
plumbs spec.params['url_ttl_seconds'] through).

ADMIN_SESSION_SECRET is reused as the signing key — it's already required
for the admin session cookie HMAC, so no new secret to manage. The dev
fallback (no secret → no URL) keeps tests + fresh installs working without
forcing operators to configure crypto on day one.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)


def _outputs_root() -> Path:
    """~/.gstack/outputs/images by default; override via $NOUS_IMAGE_OUTPUTS."""
    override = os.environ.get("NOUS_IMAGE_OUTPUTS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".gstack" / "outputs" / "images"


def _signing_key() -> bytes | None:
    secret = get_settings().ADMIN_SESSION_SECRET
    return secret.encode("utf-8") if secret else None


def _sign(uuid_str: str, expires: int) -> str:
    key = _signing_key()
    if key is None:
        return ""
    msg = f"{uuid_str}.{expires}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_token(uuid_str: str, expires: int, token: str) -> bool:
    """True iff token is a valid signature AND expires is in the future.

    Uses secrets.compare_digest to avoid leaking timing info on the
    hex digest comparison.
    """
    key = _signing_key()
    if key is None:
        return False
    if expires <= int(time.time()):
        return False
    expected = _sign(uuid_str, expires)
    return secrets.compare_digest(expected, token)


def write_image(png_bytes: bytes, *, ext: str = "png", ttl_seconds: int = 3600) -> dict:
    """Persist `png_bytes` under outputs/images/<date>/<uuid>.<ext>.

    Returns:
        {
          "uuid":  random uuid hex,
          "path":  absolute on-disk path (Path),
          "url":   signed URL (str) or None when no secret configured,
          "ext":   ext (without leading dot),
          "expires": unix timestamp the URL is valid until (int) or None,
        }
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = _outputs_root() / today
    root.mkdir(parents=True, exist_ok=True)

    file_uuid = _uuid.uuid4().hex
    path = root / f"{file_uuid}.{ext}"
    # Atomic-ish: write to .tmp then rename so partial reads can't see
    # half-flushed PNG bytes if the process crashes mid-write.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(png_bytes)
    os.replace(tmp, path)

    url: str | None = None
    expires: int | None = None
    if _signing_key() is not None:
        expires = int(time.time()) + max(60, int(ttl_seconds))
        token = _sign(file_uuid, expires)
        url = f"/files/images/{today}/{file_uuid}.{ext}?token={token}&expires={expires}"

    return {
        "uuid": file_uuid,
        "path": path,
        "url": url,
        "ext": ext,
        "expires": expires,
    }


def resolve_path(date: str, uuid_str: str, ext: str) -> Path:
    """Build the on-disk path for a given date/uuid/ext triple.

    Used by the static route after HMAC verification; never trusts the
    caller-supplied date+uuid until verify_token has succeeded.
    """
    return _outputs_root() / date / f"{uuid_str}.{ext}"
