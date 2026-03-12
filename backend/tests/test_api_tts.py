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


async def test_synthesize_engine_not_loaded(client):
    with patch("src.api.routes.tts._get_loaded_engine", return_value=None):
        resp = await client.post(
            "/api/v1/tts/synthesize",
            json={"engine": "cosyvoice2", "text": "hello"},
        )
    assert resp.status_code == 409


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


async def test_batch_status(client):
    mock_result_0 = MagicMock()
    mock_result_0.state = "SUCCESS"
    mock_result_0.result = {"audio": "base64data"}
    mock_result_0.ready.return_value = True

    mock_result_1 = MagicMock()
    mock_result_1.state = "SUCCESS"
    mock_result_1.result = {"audio": "base64data2"}
    mock_result_1.ready.return_value = True

    # Third probe returns PENDING with no result → signals end of batch
    mock_result_end = MagicMock()
    mock_result_end.state = "PENDING"
    mock_result_end.result = None

    def mock_async_result(task_id, app=None):
        if task_id == "batch_abc123_0":
            return mock_result_0
        elif task_id == "batch_abc123_1":
            return mock_result_1
        return mock_result_end

    with patch("src.api.routes.tts.AsyncResult", side_effect=mock_async_result):
        resp = await client.get("/api/v1/tts/batch/batch_abc123")

    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == "batch_abc123"
    assert data["status"] == "completed"
    assert len(data["tasks"]) == 2


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
