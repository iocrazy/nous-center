"""GPU 热保护看门狗 —— 超温自动降载 + 告警。

背景(2026-07-06 活机实测):PRO 6000 空闲就 75°C、风扇 0%;只是**加载一个模型**
温度就冲到 86°C、离降频点(~93°C)仅 ~7°C;系统却毫无察觉、不记录、不保护。审查
早已指出「温度采了却没用」。本看门狗把负载最危险的阶段(满载推理冲 600W)兜住:
温度过临界就卸掉那张卡上**可安全卸载**的模型(复用 evict_lru,尊重 resident/in_use
守卫,绝不卸正在推理的模型),并大声告警。

阈值可 env 覆盖:NOUS_GPU_TEMP_WARN(默认 85)/ NOUS_GPU_TEMP_CRITICAL(默认 90)。
"""
from __future__ import annotations

import asyncio
import logging
import os

from src.services.gpu_monitor import poll_gpu_stats

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 15.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


WARN_C = _env_int("NOUS_GPU_TEMP_WARN", 85)
CRITICAL_C = _env_int("NOUS_GPU_TEMP_CRITICAL", 90)


def thermal_action(temp: int | None, warn: int, critical: int) -> str:
    """纯决策:'ok' / 'warn' / 'shed'。temp None(读不到)→ 'ok',绝不误动作。"""
    if temp is None:
        return "ok"
    if temp >= critical:
        return "shed"
    if temp >= warn:
        return "warn"
    return "ok"


async def check_thermal(model_manager, *, warn: int = WARN_C, critical: int = CRITICAL_C) -> None:
    """扫一遍各卡温度,过阈值告警 / 降载。降载复用 evict_lru(只卸非常驻、非在用的模型)。"""
    for gpu in poll_gpu_stats():
        temp = gpu.get("temperature")
        idx = gpu.get("index")
        fan = gpu.get("fan_speed")
        # 风扇异常告警:温度到告警线但风扇 0% → 风扇可能没转/读不到(2026-07-06 活机
        # 就是这情况:加载一下就 86°C、风扇 0%)。散热失效前给运维一个明确信号。
        if temp is not None and temp >= warn and fan == 0:
            logger.warning(
                "GPU %s 温度 %s°C 已达告警线但风扇转速 0%% —— 风扇可能没转/散热失效,"
                "请立即物理检查风扇与风道!",
                idx, temp,
            )
        action = thermal_action(temp, warn, critical)
        if action == "shed":
            evicted = await model_manager.evict_lru(gpu_index=idx)
            if evicted:
                logger.critical(
                    "GPU %s 温度 %s°C ≥ %s°C 临界 → 热保护降载:卸掉 %s 救卡",
                    idx, temp, critical, evicted,
                )
            else:
                logger.critical(
                    "GPU %s 温度 %s°C ≥ %s°C 临界,但无可安全卸载的模型(都常驻/在用)"
                    " → 只能告警。请立即检查散热/风扇!",
                    idx, temp, critical,
                )
        elif action == "warn":
            logger.warning(
                "GPU %s 温度 %s°C ≥ %s°C 告警阈值,接近降频,检查散热/风扇",
                idx, temp, warn,
            )


async def gpu_thermal_guard_loop(
    model_manager, *, interval: float = POLL_INTERVAL_SECONDS,
    warn: int = WARN_C, critical: int = CRITICAL_C,
) -> None:
    """后台 loop:每 interval 秒查一次 GPU 温度。fail-soft,坏了不拖垮别的 loop。"""
    while True:
        await asyncio.sleep(interval)
        try:
            await check_thermal(model_manager, warn=warn, critical=critical)
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU thermal guard failed: %s", e)
