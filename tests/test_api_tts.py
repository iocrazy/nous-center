import pytest
from unittest.mock import patch, MagicMock
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
async def test_tts_generate_returns_task_id(client):
    with patch("src.api.routes.tts.generate_tts_task") as mock_task:
        mock_task.delay = MagicMock()
        resp = await client.post(
            "/api/v1/tts/generate",
            json={"text": "hello world"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert data["type"] == "tts"
        assert "task_id" in data
        mock_task.delay.assert_called_once()
