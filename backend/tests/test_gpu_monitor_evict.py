"""Lane 0: check_and_evict 改道到 model_manager.evict_lru 的回归测试。"""
import pytest

from src.services.gpu_monitor import check_and_evict


class _FakeModelManager:
    def __init__(self):
        self.evict_calls: list[int | None] = []

    async def evict_lru(self, gpu_index=None):
        self.evict_calls.append(gpu_index)
        return None  # 没有可驱逐的模型


@pytest.mark.asyncio
async def test_check_and_evict_calls_model_manager_evict_lru(monkeypatch):
    """GPU 低显存时，check_and_evict 调 model_manager.evict_lru(该 GPU index)。"""
    monkeypatch.setattr(
        "src.services.gpu_monitor.poll_gpu_stats",
        lambda: [{"index": 0, "free_mb": 1024, "used_mb": 23000,
                  "total_mb": 24000, "utilization_pct": 50, "temperature": 60}],
    )
    fake_mgr = _FakeModelManager()
    await check_and_evict(fake_mgr, reserved_gb=4.0)
    assert fake_mgr.evict_calls == [0], "低显存 GPU 0 应触发 evict_lru(0)"


@pytest.mark.asyncio
async def test_check_and_evict_skips_healthy_gpu(monkeypatch):
    """显存充足时不驱逐。"""
    monkeypatch.setattr(
        "src.services.gpu_monitor.poll_gpu_stats",
        lambda: [{"index": 0, "free_mb": 20000, "used_mb": 4000,
                  "total_mb": 24000, "utilization_pct": 10, "temperature": 50}],
    )
    fake_mgr = _FakeModelManager()
    await check_and_evict(fake_mgr, reserved_gb=4.0)
    assert fake_mgr.evict_calls == [], "显存充足不应 evict"
