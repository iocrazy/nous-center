from unittest.mock import patch, MagicMock


async def test_list_engines(db_client):
    resp = await db_client.get("/api/v1/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert isinstance(engines, list)
    assert len(engines) > 0
    engine = engines[0]
    assert "name" in engine
    assert "status" in engine
    assert engine["status"] in ("loaded", "unloaded")


async def test_list_engines_contains_all_tts(db_client):
    resp = await db_client.get("/api/v1/engines")
    names = {e["name"] for e in resp.json()}
    assert "cosyvoice2" in names
    assert "indextts2" in names
    assert "moss_tts" in names


async def test_list_engines_returns_metadata_fields(db_client):
    resp = await db_client.get("/api/v1/engines")
    engine = resp.json()[0]
    # New fields should be present even if null
    assert "has_metadata" in engine
    assert "local_exists" in engine
    assert "model_size" in engine
    assert "frameworks" in engine


async def test_list_engines_filter_by_type(db_client):
    resp = await db_client.get("/api/v1/engines?type=tts")
    engines = resp.json()
    assert all(e["type"] == "tts" for e in engines)

    resp2 = await db_client.get("/api/v1/engines?type=image")
    engines2 = resp2.json()
    assert all(e["type"] == "image" for e in engines2)


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
