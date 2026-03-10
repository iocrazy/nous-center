import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.workers.tts_engines.base import TTSResult

FAKE_WAV = b"RIFF" + b"\x00" * 100


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


@pytest.mark.asyncio
async def test_synthesize_returns_audio(client):
    fake_result = TTSResult(
        audio_bytes=FAKE_WAV,
        sample_rate=24000,
        duration_seconds=1.5,
        format="wav",
    )
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.synthesize.return_value = fake_result

    with patch("src.api.routes.tts._get_loaded_engine", return_value=mock_engine):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "audio_base64" in data
    assert data["engine"] == "cosyvoice2"
    assert data["duration_seconds"] == 1.5
    assert data["rtf"] > 0


@pytest.mark.asyncio
async def test_synthesize_engine_not_loaded(client):
    with patch("src.api.routes.tts._get_loaded_engine", return_value=None):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_batch_tts(client):
    mock_delay = MagicMock(return_value=MagicMock(id="fake-id"))
    mock_resolve = AsyncMock(
        return_value={"engine": "cosyvoice2", "params": {"voice": "default"}}
    )

    with (
        patch("src.api.routes.tts.generate_tts_task.delay", mock_delay),
        patch("src.api.routes.tts._resolve_preset", mock_resolve),
    ):
        resp = await client.post(
            "/api/v1/tts/batch",
            json={
                "segments": [
                    {"voice_preset": "host", "text": "Hello"},
                    {"voice_preset": "guest", "text": "Hi"},
                ]
            },
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "batch_id" in data
    assert len(data["tasks"]) == 2
    assert data["tasks"][0]["index"] == 0
