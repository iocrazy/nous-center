"""Regression: CORS exposes the headers JS clients need to read.

PR #34 emits ETag on every cached GET. PR #35 fixed JS clients getting null
when calling response.headers.get('etag') by adding ETag to the CORS
expose_headers list. If someone trims that list back to just X-Request-Id,
this test fails before it ships.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.mark.anyio
async def test_cors_exposes_etag_and_request_id():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Real cross-origin GET (Origin header makes Starlette CORS apply expose_headers)
        r = await ac.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    expose = r.headers.get("access-control-expose-headers", "")
    # Headers list is comma-separated; verify both names are present (order-insensitive)
    exposed = {h.strip() for h in expose.split(",") if h.strip()}
    assert "ETag" in exposed, f"ETag must be exposed for JS clients to read it; got: {expose!r}"
    assert "X-Request-Id" in exposed, f"X-Request-Id must stay exposed; got: {expose!r}"


@pytest.mark.anyio
async def test_cors_preflight_allows_credentials():
    """Cookie-based admin login requires credentials in CORS preflight."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.options(
            "/api/v1/engines",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    # Starlette CORS responds 200 to preflight when origin is allowed
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-credentials") == "true"
