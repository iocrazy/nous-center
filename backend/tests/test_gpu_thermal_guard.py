"""GPU 热保护看门狗 —— 超温自动降载 + 告警(活机实测:加载一下就 86°C、风扇 0%,
离降频仅 ~7°C,系统却毫无察觉/保护)。复用 evict_lru(尊重 resident/in_use 守卫)。"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.gpu_thermal_guard import thermal_action, check_thermal


class TestThermalAction:
    def test_ok_below_warn(self):
        assert thermal_action(70, 85, 90) == "ok"

    def test_warn_at_threshold(self):
        assert thermal_action(85, 85, 90) == "warn"
        assert thermal_action(88, 85, 90) == "warn"

    def test_shed_at_critical(self):
        assert thermal_action(90, 85, 90) == "shed"
        assert thermal_action(95, 85, 90) == "shed"

    def test_none_temp_is_ok(self):
        # 读不到温度 → 不误动作
        assert thermal_action(None, 85, 90) == "ok"


@pytest.mark.asyncio
async def test_check_thermal_sheds_on_critical(monkeypatch):
    """临界温度 → 调 evict_lru(该卡)降载。"""
    import src.services.gpu_thermal_guard as g
    monkeypatch.setattr(g, "poll_gpu_stats", lambda: [{"index": 1, "temperature": 92}])
    mm = MagicMock()
    mm.evict_lru = AsyncMock(return_value="qwen3_6_35b")
    await check_thermal(mm, warn=85, critical=90)
    mm.evict_lru.assert_awaited_once_with(gpu_index=1)


@pytest.mark.asyncio
async def test_check_thermal_no_shed_when_cool(monkeypatch):
    import src.services.gpu_thermal_guard as g
    monkeypatch.setattr(g, "poll_gpu_stats", lambda: [{"index": 0, "temperature": 60}])
    mm = MagicMock()
    mm.evict_lru = AsyncMock(return_value=None)
    await check_thermal(mm, warn=85, critical=90)
    mm.evict_lru.assert_not_awaited()
