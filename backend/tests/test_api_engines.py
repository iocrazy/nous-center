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


async def test_unload_clears_stale_failed_state(client):
    """round9 BUG4:load 失败残留的 'failed' 状态,unload 后必须清掉,
    否则 _build_engine_info 里它优先级高于 loaded/unloaded → GET /engines 永远显 failed。"""
    from src.api.routes import engines as engines_route

    mock_mgr = client._transport.app.state.model_manager
    mock_mgr.unload_model = AsyncMock()
    engines_route._loading_states["qwen3_tts_base"] = {
        "status": "failed", "detail": "boom",
    }

    resp = await client.post("/api/v1/engines/qwen3_tts_base/unload")
    assert resp.status_code == 200
    assert "qwen3_tts_base" not in engines_route._loading_states


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


# ----- PR-D4: 手动 unload-image-adapters 端点(image adapter 入 _models 统一字典后路径)-----


class _FakeMainMM:
    """模拟主进程 ModelManager 的 loaded_models_snapshot + unload_model(非 MagicMock,
    让 aggregate 真能 iter)。用于测 image 落在主进程(group='main')的兜底卸载路径。"""
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.unloaded: list[str] = []

    def loaded_models_snapshot(self):
        return self._snapshot

    async def unload_model(self, mid, force=False):
        self.unloaded.append(mid)
        self._snapshot = [e for e in self._snapshot if e["model_id"] != mid]


async def test_unload_image_adapters_main_fallback_unloads_locally(client, app):
    """image 落在主进程(group='main',极少)时,unload 走主进程 model_manager.unload_model,
    留 LLM/TTS 不动。runner 路径见 test_unload_image_adapters_dispatches_to_runner。"""
    app.state.runner_supervisors = []
    mm = _FakeMainMM([
        {"model_id": "image:foo:1", "model_type": "image", "gpu_index": 1, "source_files": []},
        {"model_id": "image:bar:2", "model_type": "image", "gpu_index": 2, "source_files": []},
        {"model_id": "qwen3-1.7b", "model_type": "llm", "gpu_index": 0, "source_files": []},
    ])
    app.state.model_manager = mm

    resp = await client.post("/api/v1/engines/unload-image-adapters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert set(body["unloaded"]) == {"image:foo:1", "image:bar:2"}
    assert set(mm.unloaded) == {"image:foo:1", "image:bar:2"}  # LLM 没被动
    assert "qwen3-1.7b" not in mm.unloaded


class _FakeSup:
    """模拟 RunnerSupervisor:只暴露 group_id + loaded_models(Pong 上报的快照)。"""
    def __init__(self, group_id, loaded_models):
        self.group_id = group_id
        self.loaded_models = loaded_models


async def test_image_cache_endpoint_lists_image_entries_only(client, app):
    """GET /api/v1/engines/image-cache 返 image 类已加载 adapter,LLM 不出现。

    #198 修正:image adapter 真加载在 runner 子进程,主进程 _models 恒空 —— 端点改读
    runner 经 Pong 上报、聚合到各 supervisor.loaded_models 的快照。本测试用 _FakeSup
    模拟该上报,断言只有 image 类出现、字段完整。"""
    app.state.runner_supervisors = [
        _FakeSup("image", [
            {"model_id": "image:Flux2KleinPipeline:foo:11111111", "model_type": "image",
             "gpu_index": 1, "gpu_indices": [1], "vram_mb": 19000,
             "pipeline_class": "Flux2KleinPipeline",
             "source_files": ["/m/Flux2-Klein-9B.safetensors", "/m/qwen3.safetensors", "/m/vae.safetensors"],
             "last_used_ago_sec": 3.2},
            {"model_id": "image:AnimaPipeline:bar:22222222", "model_type": "image",
             "gpu_index": 2, "gpu_indices": [2], "vram_mb": 4000,
             "pipeline_class": "AnimaPipeline", "source_files": [], "last_used_ago_sec": 1.0},
        ]),
        _FakeSup("tts", [
            {"model_id": "qwen-tts", "model_type": "tts", "gpu_index": 0,
             "gpu_indices": [0], "vram_mb": 5000, "pipeline_class": None,
             "source_files": [], "last_used_ago_sec": 0.5},
        ]),
    ]

    resp = await client.get("/api/v1/engines/image-cache")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2  # tts 不算
    ids = {e["model_id"] for e in body["entries"]}
    assert ids == {
        "image:Flux2KleinPipeline:foo:11111111",
        "image:AnimaPipeline:bar:22222222",
    }
    # 字段完整 — gpu_index / pipeline_class / vram_mb / source_files / group_id
    e0 = next(e for e in body["entries"] if "Flux2" in e["model_id"])
    assert e0["gpu_index"] == 1
    assert e0["pipeline_class"] == "Flux2KleinPipeline"
    assert e0["vram_mb"] == 19000
    assert e0["group_id"] == "image"
    assert "/m/Flux2-Klein-9B.safetensors" in e0["source_files"]


async def test_loaded_adapters_endpoint_lists_runner_combo_entities(client, app):
    """Bug 3 PR-2c:GET /loaded-adapters 列 runner 里的 combo adapter 实体(image+tts),
    带 display_name(源文件 basename);group='main' 不进(那以注册卡形式在 engines 列表)。"""
    app.state.runner_supervisors = [
        _FakeSup("image", [
            {"model_id": "image:foo:1", "model_type": "image", "gpu_index": 1,
             "vram_mb": 19000, "pipeline_class": "Flux2KleinPipeline",
             "source_files": ["/m/Flux2-Klein-9B.safetensors", "/m/clip.safetensors"],
             "last_used_ago_sec": 3.4},
        ]),
        _FakeSup("tts", [
            {"model_id": "tts:bar:2", "model_type": "tts", "gpu_index": 2,
             "vram_mb": 5000, "pipeline_class": None, "source_files": [],
             "last_used_ago_sec": 1.0},
        ]),
    ]
    resp = await client.get("/api/v1/engines/loaded-adapters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    by_id = {e["model_id"]: e for e in body["entries"]}
    assert by_id["image:foo:1"]["display_name"] == "Flux2-Klein-9B.safetensors"
    assert by_id["image:foo:1"]["group_id"] == "image"
    assert by_id["tts:bar:2"]["display_name"] == "tts:bar:2"  # 无 source → 退回 model_id


def test_explain_image_combo_key_unpacks_all_components():
    """_explain_image_combo_key:cache miss 日志要把 5 个字段拆开人能读。
    PR-D5 诊断字段稳定性用 — 直接读 backend log 比 sha256 hash 易诊断 100×。"""
    from src.services.model_manager import ModelManager

    combo = (
        "Flux2KleinPipeline",
        "none",
        ("/m/flux2.safetensors", "cuda:1", "bfloat16", frozenset()),
        ("/m/qwen3.safetensors", "cuda:1", "bfloat16", frozenset()),
        ("/m/vae.safetensors", "cuda:1", "bfloat16", frozenset({("turbo", 0.8)})),
    )
    out = ModelManager._explain_image_combo_key(combo)
    assert out["pipeline_class"] == "Flux2KleinPipeline"
    assert out["offload"] == "none"
    assert out["transformer"]["file"] == "/m/flux2.safetensors"
    assert out["transformer"]["dtype"] == "bfloat16"
    assert out["vae"]["loras"] == ["turbo@0.8"]


async def test_unload_image_adapters_when_none_returns_zero(client, app):
    """空状态下也要 200 + count=0(让前端按钮即使没 image 时不报错)。"""
    app.state.runner_supervisors = []
    app.state.model_manager._models = {}
    resp = await client.post("/api/v1/engines/unload-image-adapters")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


class _FakeClient:
    def __init__(self):
        self.unloaded: list[str] = []

    async def unload_model(self, mid):
        self.unloaded.append(mid)


class _FakeSupWithClient:
    def __init__(self, group_id, loaded_models):
        self.group_id = group_id
        self.loaded_models = loaded_models
        self.client = _FakeClient()

    async def _reconcile_loaded(self):
        self.loaded_models = []  # 模拟 runner 卸载后快照清空


async def test_unload_image_adapters_dispatches_to_runner(client, app):
    """Bug 3 PR-2a:image adapter 在 runner 子进程,unload 必须派 UnloadModel 给 runner
    (非 no-op),并卸载后 reconcile 快照。"""
    sup = _FakeSupWithClient("image", [
        {"model_id": "image:foo:1", "model_type": "image", "gpu_index": 1, "source_files": []},
        {"model_id": "image:bar:2", "model_type": "image", "gpu_index": 2, "source_files": []},
    ])
    app.state.runner_supervisors = [sup]
    resp = await client.post("/api/v1/engines/unload-image-adapters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert set(sup.client.unloaded) == {"image:foo:1", "image:bar:2"}  # 真派给 runner
    assert sup.loaded_models == []  # reconcile 被调用 → 快照刷新


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
