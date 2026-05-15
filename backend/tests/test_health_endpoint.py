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
