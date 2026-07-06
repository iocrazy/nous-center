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
    """Gate /api/* when ANY admin credential is configured.

    Either the browser password (cookie login) or a CLI ADMIN_TOKEN arms the
    gate — a token-only headless deployment must still be protected, and the
    route guard + middleware must agree on when auth is on. Both empty = dev mode.
    """
    s = get_settings()
    return bool(s.ADMIN_PASSWORD or s.ADMIN_TOKEN)


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
    try:
        secret = _secret()
    except RuntimeError:
        # token-only 部署(ADMIN_SESSION_SECRET 空):无法验签 cookie token → 视为无效,
        # 不让含 "." 的伪造 cookie 触发未捕获 500(S3 robustness)。
        return False
    expected = hmac.new(secret, expires_str.encode("ascii"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def admin_token_matches(authorization: str | None) -> bool:
    """CLI/curl path: ``Authorization: Bearer <ADMIN_TOKEN>``.

    Only valid when an ADMIN_TOKEN is configured (constant-time compare). This is
    the second accepted credential alongside the browser session cookie, so the
    documented CLI bearer actually clears the ``/api/*`` gate instead of being
    blocked by the cookie middleware before its route ever runs.
    """
    token = get_settings().ADMIN_TOKEN
    if not token or not authorization or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:], token)


def request_is_authed(request: Request) -> bool:
    if not is_login_required():
        return True
    if verify_token(request.cookies.get(COOKIE_NAME)):
        return True
    return admin_token_matches(request.headers.get("Authorization"))


def websocket_is_authed(websocket: WebSocket) -> bool:
    if not is_login_required():
        return True
    if verify_token(websocket.cookies.get(COOKIE_NAME)):
        return True
    return admin_token_matches(websocket.headers.get("Authorization"))
