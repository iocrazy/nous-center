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
    # Database may not be available in test; accept both "ok" and "degraded"
    assert resp.json()["status"] in ("ok", "degraded")


# V1' P1: legacy /api/v1/models handler (formerly in tasks.py) was removed
# in favor of the new scanner-backed endpoint in routes/models.py. Its
# replacement is covered by test_model_scanner.py.
