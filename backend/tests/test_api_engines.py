from unittest.mock import patch, MagicMock, AsyncMock


async def test_list_engines(db_client):
    with patch("src.api.routes.engines.scan_local_models", return_value={"tts/cosyvoice2-0.5b", "tts/indextts-2", "tts/moss-tts"}):
        resp = await db_client.get("/api/v1/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert isinstance(engines, list)
    assert len(engines) > 0
    engine = engines[0]
    assert "name" in engine
    assert "status" in engine
    assert engine["status"] in ("loaded", "unloaded")


async def test_list_engines_includes_all(db_client):
    """All engines are returned regardless of local availability."""
    with patch("src.api.routes.engines.scan_local_models", return_value={"tts/cosyvoice2-0.5b"}):
        resp = await db_client.get("/api/v1/engines")
    engines = resp.json()
    names = {e["name"] for e in engines}
    assert "cosyvoice2" in names
    # sdxl also appears now (with local_exists=False)
    assert "sdxl" in names
    # cosyvoice2 has local_exists=True, sdxl has local_exists=False
    by_name = {e["name"]: e for e in engines}
    assert by_name["cosyvoice2"]["local_exists"] is True
    assert by_name["sdxl"]["local_exists"] is False


async def test_list_engines_returns_metadata_fields(db_client):
    with patch("src.api.routes.engines.scan_local_models", return_value={"tts/cosyvoice2-0.5b"}):
        resp = await db_client.get("/api/v1/engines")
    engine = resp.json()[0]
    assert "has_metadata" in engine
    assert "local_exists" in engine
    assert "model_size" in engine
    assert "frameworks" in engine


async def test_list_engines_filter_by_type(db_client):
    local = {"tts/cosyvoice2-0.5b", "tts/indextts-2", "tts/moss-tts"}
    with patch("src.api.routes.engines.scan_local_models", return_value=local):
        resp = await db_client.get("/api/v1/engines?type=tts")
    engines = resp.json()
    assert all(e["type"] == "tts" for e in engines)
    assert len(engines) > 0


async def test_load_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/load")
    assert resp.status_code == 404


async def test_unload_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/unload")
    assert resp.status_code == 404


async def test_load_engine_success(client):
    with patch("src.api.routes.engines.model_scheduler.load_model", new_callable=AsyncMock) as mock_load:
        resp = await client.post("/api/v1/engines/cosyvoice2/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "cosyvoice2"
    assert data["status"] == "loaded"
    mock_load.assert_called_once_with("cosyvoice2")


async def test_unload_resident_engine_rejected(client):
    resp = await client.post("/api/v1/engines/cosyvoice2/unload")
    assert resp.status_code == 409


async def test_scheduler_status(client):
    resp = await client.get("/api/v1/engines/scheduler/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "loaded" in data
    assert "references" in data
    assert "last_used" in data
