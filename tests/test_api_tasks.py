import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_models_endpoint(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu_status" in data
    assert "models" in data
