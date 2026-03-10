import pytest
from unittest.mock import patch, AsyncMock
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
async def test_understand_image(client):
    mock_response = {"text": "A cat sitting on a table", "model": "qwen25-vl"}
    with patch("src.api.routes.understand.call_vllm", new_callable=AsyncMock) as mock:
        mock.return_value = mock_response
        resp = await client.post(
            "/api/v1/understand/image",
            json={"image_url": "/path/to/img.png", "question": "What is this?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "A cat sitting on a table"
