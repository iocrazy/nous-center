"""Admin browser session — HMAC-signed cookie token.

Format: ``<expires_unix>.<hex_hmac_sha256>``. Verification compares the HMAC
with ``hmac.compare_digest`` and rejects expired tokens. Used by ``/sys/admin/login``
and the request middleware that gates ``/api/*`` + ``/ws/*``.
"""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Request, WebSocket

from src.config import get_settings

COOKIE_NAME = "nous_admin_session"


def is_login_required() -> bool:
    return bool(get_settings().ADMIN_PASSWORD)


def _secret() -> bytes:
    s = get_settings()
    if not s.ADMIN_SESSION_SECRET:
        # Refuse to silently sign with an empty key when password gate is on —
        # otherwise anyone could forge a token by guessing "".
        raise RuntimeError("ADMIN_SESSION_SECRET must be set when ADMIN_PASSWORD is set")
    return s.ADMIN_SESSION_SECRET.encode("utf-8")


def issue_token(now: int | None = None) -> tuple[str, int]:
    """Return (token, max_age_seconds)."""
    s = get_settings()
    now = now if now is not None else int(time.time())
    expires = now + s.ADMIN_SESSION_MAX_AGE_SECONDS
    sig = hmac.new(_secret(), str(expires).encode("ascii"), hashlib.sha256).hexdigest()
    return f"{expires}.{sig}", s.ADMIN_SESSION_MAX_AGE_SECONDS


def verify_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expires_str, sig = token.split(".", 1)
    if not expires_str.isdigit():
        return False
    expires = int(expires_str)
    if expires < int(time.time()):
        return False
    expected = hmac.new(_secret(), expires_str.encode("ascii"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def request_is_authed(request: Request) -> bool:
    if not is_login_required():
        return True
    return verify_token(request.cookies.get(COOKIE_NAME))


def websocket_is_authed(websocket: WebSocket) -> bool:
    if not is_login_required():
        return True
    return verify_token(websocket.cookies.get(COOKIE_NAME))
