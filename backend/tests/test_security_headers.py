"""二轮安全 §7:所有响应带零风险安全头。"""
import pytest


@pytest.mark.asyncio
async def test_security_headers_present():
    from httpx import ASGITransport, AsyncClient
    from src.api.main import create_app
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("referrer-policy") == "same-origin"
