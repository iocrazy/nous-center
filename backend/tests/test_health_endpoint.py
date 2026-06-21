"""Lane H: /health 端点扩展测试 —— load_failures + runners + degraded 状态。"""
import pytest

from src.api.main import create_app


@pytest.mark.asyncio
async def test_health_has_load_failures_and_runners_keys():
    """/health 返回体含 load_failures 和 runners 两个新字段。"""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "load_failures" in body
    assert "runners" in body
    assert isinstance(body["load_failures"], dict)
    assert isinstance(body["runners"], list)
    # 启动提示块(c):resident 加载进度;mgr 替身无 _registry → 降级 0/0 不崩(回归守卫)。
    assert "startup" in body
    assert set(body["startup"]) >= {"resident_total", "resident_loaded", "preloading"}


@pytest.mark.asyncio
async def test_health_no_runners_when_supervisors_unset():
    """app.state.runner_supervisors 未设置（Lane A 还没接入）→ runners 是空列表，不报错。"""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["runners"] == []


@pytest.mark.asyncio
async def test_health_degraded_when_load_failure_present():
    """mm._load_failures 非空 → status 是 'degraded'，failure 内容出现在 load_failures。"""
    from types import SimpleNamespace

    from httpx import ASGITransport, AsyncClient

    app = create_app()
    # ASGITransport 默认不跑 lifespan，conftest 通常 mock model_manager —— 这里
    # 注入一个最小 mock，含 loaded_model_ids + _load_failures。
    app.state.model_manager = SimpleNamespace(
        loaded_model_ids=[],
        _load_failures={"flux2-dev": "OutOfMemoryError: CUDA out of memory"},
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["load_failures"]["flux2-dev"] == "OutOfMemoryError: CUDA out of memory"


@pytest.mark.asyncio
async def test_health_reports_runner_snapshot():
    """app.state.runner_supervisors 有 supervisor → runners 列表含其 health_snapshot。"""
    from httpx import ASGITransport, AsyncClient

    class _FakeSupervisor:
        def health_snapshot(self):
            return {
                "group_id": "image", "gpus": [2], "running": False,
                "restart_count": 2, "pid": None,
            }

    app = create_app()
    app.state.runner_supervisors = [_FakeSupervisor()]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runners"] == [{
        "group_id": "image", "gpus": [2], "running": False,
        "restart_count": 2, "pid": None,
    }]
    # runner 不 running → status degraded
    assert body["status"] == "degraded"


class _FakeSession:
    """让 /health 的 `SELECT 1` 库检查通过,隔离出 runner-healthy 逻辑(否则测试环境
    无 DB → database=error → degraded 盖过被测项)。"""
    async def execute(self, *a, **k):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_db_ok(monkeypatch):
    monkeypatch.setattr(
        "src.models.database.get_session_factory", lambda: (lambda: _FakeSession())
    )


async def _health_status_with_llm_runner(monkeypatch, snapshot: dict) -> dict:
    from httpx import ASGITransport, AsyncClient

    _patch_db_ok(monkeypatch)

    class _LLMRunner:
        def health_snapshot(self):
            return snapshot

    app = create_app()
    app.state.llm_runner = _LLMRunner()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_health_idle_llm_runner_not_degraded(monkeypatch):
    """LLMRunner 稳定停在 IDLE(running=False 但 healthy=True)—— vLLM 由 model_mgr
    懒加载/常驻预载,不经 LLMRunner 自 spawn。/health 不该因此误报 degraded。
    回归:旧逻辑用 `not running` → 永久 degraded → 公开状态页误显黄灯。"""
    body = await _health_status_with_llm_runner(monkeypatch, {
        "group_id": "llm", "gpus": [1], "running": False,
        "healthy": True, "restart_count": 0, "pid": None, "current_task": None,
    })
    assert body["runners"][0]["healthy"] is True
    assert body["runners"][0]["running"] is False
    assert body["status"] != "degraded"


@pytest.mark.asyncio
async def test_health_failed_llm_runner_degraded(monkeypatch):
    """LLMRunner FAILED(healthy=False)→ degraded(真故障要报出来)。"""
    body = await _health_status_with_llm_runner(monkeypatch, {
        "group_id": "llm", "gpus": [1], "running": False,
        "healthy": False, "restart_count": 1, "pid": None, "current_task": None,
    })
    assert body["status"] == "degraded"
