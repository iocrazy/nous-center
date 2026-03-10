import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.api.websocket import ConnectionManager


def test_connection_manager_init():
    manager = ConnectionManager()
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_endpoint():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Just verify the websocket route is registered
        routes = [r.path for r in app.routes]
        assert "/ws/tasks/{task_id}" in routes
