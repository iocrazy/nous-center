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
async def test_generate_image_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/image",
            json={"prompt": "a cat"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["task_id"] == "fake-task-id"
        assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_generate_tts_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/tts",
            json={"text": "hello world"},
        )
        assert resp.status_code == 202


@pytest.mark.asyncio
async def test_generate_video_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/video",
            json={"prompt": "sunset timelapse"},
        )
        assert resp.status_code == 202
