import pytest
from httpx import AsyncClient
from unittest.mock import patch

pytestmark = pytest.mark.anyio

async def test_monitor_stats(db_client: AsyncClient):
    mock_gpu_stats = [
        {
            "index": 0,
            "name": "Mock GPU",
            "utilization_gpu": 10,
            "utilization_memory": 16.3,
            "temperature": 40,
            "fan_speed": 0,
            "power_draw_w": 50.0,
            "power_limit_w": 300.0,
            "memory_used_mb": 4000,
            "memory_total_mb": 24576,
            "memory_free_mb": 20576,
            "processes": [],
        },
    ]
    with patch("src.api.routes.monitor._gpu_stats_nvidia_smi", return_value=mock_gpu_stats), \
         patch("src.api.routes.monitor._gpu_processes", return_value={}), \
         patch("src.api.routes.monitor._top_processes", return_value=[]), \
         patch("src.services.gpu_monitor.DEFAULT_RESERVED_GB", 4.0):
        resp = await db_client.get("/api/v1/monitor/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpus" in data
    assert "system" in data
