# tests/test_api_cors.py
# Uses shared `client` fixture from conftest.py


async def test_cors_preflight(client):
    resp = await client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


async def test_cors_allows_console_origin(client):
    resp = await client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
