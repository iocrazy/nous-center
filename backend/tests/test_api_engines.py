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
    """Endpoint kicks off background load and returns 'loading' immediately.
    The background task eventually calls model_manager.load_model."""
    import asyncio
    from src.api.routes import engines as engines_route

    mock_mgr = client._transport.app.state.model_manager
    mock_mgr.load_model = AsyncMock()
    mock_mgr.is_loaded = MagicMock(return_value=False)
    # Reset loading-state cache so prior tests can't poison this one
    engines_route._loading_states.pop("cosyvoice2", None)

    resp = await client.post("/api/v1/engines/cosyvoice2/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "cosyvoice2"
    assert data["status"] == "loading"

    # Yield to the loop so the background task can run + await load_model
    for _ in range(10):
        await asyncio.sleep(0.01)
        if mock_mgr.load_model.await_count > 0:
            break
    mock_mgr.load_model.assert_awaited_once_with("cosyvoice2")


async def test_unload_non_loaded_engine(client):
    """Unloading a non-loaded engine should succeed (no-op)."""
    resp = await client.post("/api/v1/engines/qwen3_tts_base/unload")
    assert resp.status_code == 200


async def test_load_rejects_engine_without_adapter(client, monkeypatch):
    """Auto-detected diffusers (no adapter) must 422 with a config hint
    instead of starting a background task that ValueErrors. Pre-fix the
    user saw a misleading 'failed' badge with no path forward."""
    from src.api.routes import engines as engines_route

    monkeypatch.setattr(engines_route, "scan_models", lambda: {
        "ernie_image": {
            "name": "ernie_image", "type": "image", "vram_gb": 35.3,
            "resident": False, "local_path": "image/diffusers/ERNIE-Image",
            "auto_detected": True,
            # No adapter — this is the case we're guarding.
        },
    })
    resp = await client.post("/api/v1/engines/ernie_image/load")
    assert resp.status_code == 422
    assert "adapter" in resp.text.lower()


async def test_scheduler_status(client):
    resp = await client.get("/api/v1/engines/scheduler/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "loaded" in data
    assert "references" in data
    assert "last_used" in data


# ----- PR-11: scan endpoint 拆分「识别 / 本地可用 / 未下载」-----


async def test_scan_endpoint_returns_local_available_split(client, monkeypatch):
    """yaml 配 3 个 / 本地仅 2 个 → count=3, local_available=2, not_local=1。

    用户报告:scan toast 显示「扫描完成 25 个」但引擎库只显示 16,差异是
    yaml 配但没下载到磁盘的模型(被 list_all_engines `local_path not in
    local_dirs` 过滤)。本测试钉死 scan 接口同时返回两个数字。
    """
    from src.api.routes import engines as engines_route

    monkeypatch.setattr(engines_route, "scan_models", lambda: {
        "a": {"name": "a", "type": "llm", "local_path": "llm/a"},
        "b": {"name": "b", "type": "tts", "local_path": "tts/b"},
        "c": {"name": "c", "type": "image", "local_path": "image/diffusers/c"},
    })
    # 只 a 和 b 实际下载到磁盘了
    monkeypatch.setattr(engines_route, "scan_local_models", lambda: {"llm/a", "tts/b"})

    resp = await client.post("/api/v1/engines/scan")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert data["local_available"] == 2
    assert data["not_local"] == 1
    assert set(data["models"]) == {"a", "b", "c"}


async def test_scan_endpoint_no_missing_when_all_local(client, monkeypatch):
    """全部本地有 → not_local=0,前端 toast 走老的简单文案分支。"""
    from src.api.routes import engines as engines_route

    monkeypatch.setattr(engines_route, "scan_models", lambda: {
        "a": {"name": "a", "type": "llm", "local_path": "llm/a"},
    })
    monkeypatch.setattr(engines_route, "scan_local_models", lambda: {"llm/a"})

    resp = await client.post("/api/v1/engines/scan")
    data = resp.json()
    assert data["count"] == 1
    assert data["local_available"] == 1
    assert data["not_local"] == 0
