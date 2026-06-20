"""The admin session gate covers /api/* — but the Ollama-compat external
endpoints (/api/chat, /api/generate, /api/tags, /api/show) live under /api/
because the Ollama protocol fixes those paths. They authenticate with the user's
API key in-route, so the admin cookie gate must let them through; otherwise every
external Ollama client gets 401 before its route runs.

Regression: this was silently broken in production (ADMIN_PASSWORD set → gate on
→ /api/chat returned "admin login required").
"""

from __future__ import annotations

import pytest
from starlette.requests import Request

from src.api import admin_session
from src.api.middleware import AdminSessionGateMiddleware


def _req(path: str) -> Request:
    return Request({"type": "http", "path": path, "method": "POST", "headers": []})


async def _call_next(_req):
    return "PASSED_THROUGH"


@pytest.fixture
def gate_on_no_creds(monkeypatch):
    # Simulate: admin gate armed, request carries no valid credential.
    monkeypatch.setattr(admin_session, "request_is_authed", lambda r: False)
    return AdminSessionGateMiddleware(app=lambda scope, receive, send: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/api/tags", "/api/show"])
async def test_ollama_paths_bypass_admin_gate(gate_on_no_creds, path):
    result = await gate_on_no_creds.dispatch(_req(path), _call_next)
    assert result == "PASSED_THROUGH"


@pytest.mark.asyncio
async def test_management_path_still_blocked(gate_on_no_creds):
    resp = await gate_on_no_creds.dispatch(_req("/api/v1/engines"), _call_next)
    assert getattr(resp, "status_code", None) == 401


@pytest.mark.asyncio
async def test_exempt_list_matches_ollama_routes(gate_on_no_creds):
    """Guard against drift: the exempt set must equal the Ollama router's
    /api/* POST/GET paths."""
    import src.api.routes.ollama_compat as oc
    from src.api.middleware import _ADMIN_GATE_EXEMPT_PATHS

    declared = {
        r.path for r in oc.router.routes if getattr(r, "path", "").startswith("/api/")
    }
    assert declared == set(_ADMIN_GATE_EXEMPT_PATHS), (
        f"ollama routes {declared} != exempt {set(_ADMIN_GATE_EXEMPT_PATHS)}"
    )
