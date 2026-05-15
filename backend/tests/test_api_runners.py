"""Lane H: GET /api/v1/monitor/runners 端点测试 —— 给前端供 per-runner 状态。"""
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.mark.asyncio
async def test_runners_endpoint_empty_when_supervisors_unset():
    """app.state.runner_supervisors 未设置（Lane A 还没接入）→ 返回空列表，200 不报错。"""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/monitor/runners")
    assert resp.status_code == 200
    assert resp.json() == {"runners": []}


@pytest.mark.asyncio
async def test_runners_endpoint_reports_supervisor_snapshots():
    """有 supervisor → runners 列表含其 health_snapshot。"""
    class _FakeSupervisor:
        def health_snapshot(self):
            return {
                "group_id": "image", "gpus": [2], "running": True,
                "restart_count": 0, "pid": 12345,
            }

    app = create_app()
    app.state.runner_supervisors = [_FakeSupervisor()]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/monitor/runners")
    assert resp.status_code == 200
    assert resp.json() == {"runners": [{
        "group_id": "image", "gpus": [2], "running": True,
        "restart_count": 0, "pid": 12345,
    }]}
