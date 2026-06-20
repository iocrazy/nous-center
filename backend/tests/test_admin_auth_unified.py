"""Unified admin auth: the route guard (require_admin) and the global
AdminSessionGateMiddleware must accept the SAME credentials — a valid admin
session cookie OR `Authorization: Bearer <ADMIN_TOKEN>` — with one source of
truth (`request_is_authed`).

Regression target: `require_admin` used to bypass whenever ADMIN_TOKEN was empty,
which in production (ADMIN_PASSWORD set, ADMIN_TOKEN empty) made it a silent
no-op. And the cookie middleware never accepted the documented CLI bearer, so
that path was dead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api import admin_session
from src.api.deps_admin import require_admin


def _make_request(headers: dict | None = None, cookies: dict | None = None) -> Request:
    raw: list[tuple[bytes, bytes]] = []
    for k, v in (headers or {}).items():
        raw.append((k.lower().encode(), v.encode()))
    if cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw.append((b"cookie", cookie_str.encode()))
    return Request({"type": "http", "headers": raw})


def _settings(password: str, secret: str, token: str):
    return SimpleNamespace(
        ADMIN_PASSWORD=password,
        ADMIN_SESSION_SECRET=secret,
        ADMIN_TOKEN=token,
        ADMIN_SESSION_MAX_AGE_SECONDS=3600,
    )


@pytest.fixture
def gated(monkeypatch):
    """Production-like: password gate ON, a CLI token configured."""
    s = _settings("hunter2", "sign-key", "cli-secret-token")
    monkeypatch.setattr(admin_session, "get_settings", lambda: s)
    return s


def test_dev_mode_passes_without_creds(monkeypatch):
    """No ADMIN_PASSWORD → gate off → anything passes (local dev)."""
    s = _settings("", "", "")
    monkeypatch.setattr(admin_session, "get_settings", lambda: s)
    assert admin_session.request_is_authed(_make_request()) is True


def test_gated_rejects_no_creds(gated):
    assert admin_session.request_is_authed(_make_request()) is False


def test_gated_accepts_valid_cookie(gated):
    token, _ = admin_session.issue_token()
    req = _make_request(cookies={admin_session.COOKIE_NAME: token})
    assert admin_session.request_is_authed(req) is True


def test_gated_accepts_admin_token_bearer(gated):
    req = _make_request(headers={"Authorization": "Bearer cli-secret-token"})
    assert admin_session.request_is_authed(req) is True


def test_gated_rejects_wrong_bearer(gated):
    req = _make_request(headers={"Authorization": "Bearer nope"})
    assert admin_session.request_is_authed(req) is False


def test_bearer_ignored_when_no_admin_token_configured(monkeypatch):
    """Password gate on but no CLI token set (current prod) → bearer can't pass,
    only the cookie does. Empty ADMIN_TOKEN must never match an empty bearer."""
    s = _settings("hunter2", "sign-key", "")
    monkeypatch.setattr(admin_session, "get_settings", lambda: s)
    assert admin_session.admin_token_matches("Bearer ") is False
    assert admin_session.admin_token_matches(None) is False


@pytest.mark.asyncio
async def test_require_admin_delegates_to_predicate(gated):
    # No creds → 401 (previously this silently passed because the bypass keyed
    # off ADMIN_TOKEN, not the actual gate state).
    with pytest.raises(HTTPException) as ei:
        await require_admin(_make_request())
    assert ei.value.status_code == 401

    # Valid CLI bearer → passes.
    ok = _make_request(headers={"Authorization": "Bearer cli-secret-token"})
    assert await require_admin(ok) is None


@pytest.mark.asyncio
async def test_require_admin_dev_mode_passes(monkeypatch):
    s = _settings("", "", "")
    monkeypatch.setattr(admin_session, "get_settings", lambda: s)
    assert await require_admin(_make_request()) is None
