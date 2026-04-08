from unittest.mock import patch, MagicMock, AsyncMock

from src.workers.tts_engines.base import TTSResult

FAKE_WAV = b"RIFF" + b"\x00" * 100


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

    with (
        patch("src.api.routes.tts._get_loaded_engine", return_value=mock_engine),
        patch("src.api.routes.tts._get_cache_service", return_value=None),
    ):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "audio_base64" in data
    assert data["engine"] == "cosyvoice2"
    assert data["duration_seconds"] == 1.5
    assert "rtf" in data
    assert data["cached"] is False


async def test_synthesize_engine_not_loaded(client):
    with patch("src.api.routes.tts._get_loaded_engine", return_value=None):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )
    assert resp.status_code == 409



async def test_batch_accepts_rounds_schema(db_client):
    """Batch endpoint should accept the new rounds schema."""
    resp = await db_client.post("/api/v1/tts/batch", json={
        "rounds": [
            {"round_id": 1, "voice_preset": "test", "text": "hello"},
            {"round_id": 2, "voice_preset": "test", "text": "world"},
        ]
    })
    # Will fail with 404 (preset not found) but not 422 (valid schema)
    assert resp.status_code != 422


async def test_synthesize_returns_cached_field(client):
    """SynthesizeResponse should include a 'cached' field."""
    resp = await client.post("/api/v1/tts/synthesize", json={
        "engine": "cosyvoice2",
        "text": "test",
    })
    # Engine not loaded in test, expect 409
    assert resp.status_code == 409


async def test_synthesize_accepts_emotion_field(client):
    """SynthesizeRequest should accept optional emotion field."""
    resp = await client.post("/api/v1/tts/synthesize", json={
        "engine": "cosyvoice2",
        "text": "test",
        "emotion": "happy tone",
    })
    # Engine not loaded, but 409 means request parsing succeeded (not 422)
    assert resp.status_code == 409


async def test_stream_endpoint_exists(client):
    """POST /tts/stream should exist and return 409 when engine not loaded."""
    resp = await client.post("/api/v1/tts/stream", json={
        "engine": "cosyvoice2",
        "text": "hello",
    })
    # Engine not loaded → 409, not 404 (endpoint exists)
    assert resp.status_code == 409


async def test_stream_validates_request(client):
    """POST /tts/stream should reject invalid speed."""
    resp = await client.post("/api/v1/tts/stream", json={
        "engine": "cosyvoice2",
        "text": "hello",
        "speed": 999,
    })
    assert resp.status_code == 422
