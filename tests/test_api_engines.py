from unittest.mock import patch, MagicMock


async def test_list_engines(client):
    resp = await client.get("/api/v1/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert isinstance(engines, list)
    assert len(engines) > 0
    engine = engines[0]
    assert "name" in engine
    assert "status" in engine
    assert engine["status"] in ("loaded", "unloaded")


async def test_list_engines_contains_all_tts(client):
    resp = await client.get("/api/v1/engines")
    names = {e["name"] for e in resp.json()}
    assert "cosyvoice2" in names
    assert "indextts2" in names
    assert "moss_tts" in names


async def test_load_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/load")
    assert resp.status_code == 404


async def test_unload_unknown_engine(client):
    resp = await client.post("/api/v1/engines/nonexistent/unload")
    assert resp.status_code == 404


async def test_load_engine_success(client):
    mock_engine = MagicMock()
    mock_engine.is_loaded = False
    with patch("src.api.routes.engines.get_engine", return_value=mock_engine):
        resp = await client.post("/api/v1/engines/cosyvoice2/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "cosyvoice2"
    assert data["status"] == "loaded"
    mock_engine.load.assert_called_once()


async def test_unload_resident_engine_rejected(client):
    resp = await client.post("/api/v1/engines/cosyvoice2/unload")
    assert resp.status_code == 409
