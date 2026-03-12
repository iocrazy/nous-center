from unittest.mock import patch, MagicMock


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


async def test_generate_video_returns_task_id(client):
    with patch("src.api.routes.generate.dispatch_task") as mock:
        mock.return_value = "fake-task-id"
        resp = await client.post(
            "/api/v1/generate/video",
            json={"prompt": "sunset timelapse"},
        )
        assert resp.status_code == 202
