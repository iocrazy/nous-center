"""vLLM 健康看门狗(2026-06-16 稳定性加固)。

周期巡检已加载的 vLLM 实例(LLM / embedding):若某模型 model_manager 记着 `loaded`
但其 vLLM 端口连不上(子进程死了 / 被 host OOM 杀了 / 改 cap·重启周期带走),自动
reconcile —— 修掉「UI 显示 loaded、调用却 ConnectError」的陈旧/孤儿状态(2026-06-16
qwen3_6_35b_a3b_fp8 真机踩到)。

判据故意保守,避免误杀健康但繁忙的实例:
  - 只对 **ConnectError**(端口连不上 = 进程真死)动作;**ReadTimeout**(vLLM 忙)不算死。
  - 连续 `fail_threshold` 个周期都确认死透才动手,跨过 startup / 正常 restart 的瞬时窗口。
reconcile 策略:
  - resident 模型 → force-unload 清陈旧态 + reload 起全新 vLLM(它本该常驻在线)。
  - 非 resident 模型 → 只 force-unload 清陈旧态(不擅自复活一个用户没钉常驻的临时模型)。

后台 loop 经 main.py lifespan 的 bg_tasks 挂载;`NOUS_DISABLE_BG_TASKS=1`(测试)或
`NOUS_DISABLE_VLLM_WATCHDOG=1` 可关闭。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.services.vllm_metrics import snapshot_all

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 45.0
DEFAULT_FAIL_THRESHOLD = 2
# 「死透」= 端口连不上。ReadTimeout/HTTP 5xx 是「活着但忙/暖机」,不 reconcile。
DEAD_ERRORS = frozenset({"ConnectError"})


def llm_targets(model_manager) -> list[tuple[str, int]]:
    """已加载且有 vLLM 端口的实例 (model_id, port) —— 与 observability/vllm 同口径。

    port 存在 ⟺ vLLM 后端(LLM 或 embedding pooling);image/tts runner 走 IPC 无此端口,
    天然不在巡检范围。"""
    out: list[tuple[str, int]] = []
    for mid, entry in getattr(model_manager, "_models", {}).items():
        adapter = getattr(entry, "adapter", None)
        if not getattr(adapter, "is_loaded", False):
            continue
        port = getattr(adapter, "_port", None) or getattr(adapter, "port", None)
        if port:
            out.append((mid, int(port)))
    return out


def _is_resident(model_manager, mid: str) -> bool:
    entry = getattr(model_manager, "_models", {}).get(mid)
    return bool(getattr(getattr(entry, "spec", None), "resident", False))


async def reconcile_dead(model_manager, mid: str) -> str:
    """清掉死掉实例的陈旧状态;resident 的再拉起。返回采取的动作字符串。"""
    resident = _is_resident(model_manager, mid)
    # force=True:陈旧的 resident 模型普通 unload 会被常驻守卫拒掉(2026-06-16 真机坐实)。
    await model_manager.unload_model(mid, force=True)
    if resident:
        await model_manager.load_model(mid)
        return "reloaded"
    return "unloaded"


async def run_cycle(
    model_manager,
    fails: dict[str, int],
    *,
    fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
    snapshot_fn: Callable[[list[tuple[str, int]]], Awaitable[list[dict]]] = snapshot_all,
    reconcile_fn: Callable[[object, str], Awaitable[str]] = reconcile_dead,
    notify: Callable[[str, str], Awaitable[None]] | None = None,
) -> list[tuple[str, str]]:
    """跑一轮巡检。`fails` 跨轮持有每模型连续失败计数(原地更新)。返回本轮 reconcile 的
    (model_id, action) 列表。抽成纯函数便于单测(loop 只是反复调它)。"""
    targets = llm_targets(model_manager)
    live = {mid for mid, _ in targets}
    # 已不在加载列表的模型清掉计数,避免 fails 无限增长 / 旧状态泄漏。
    for mid in list(fails):
        if mid not in live:
            fails.pop(mid, None)
    if not targets:
        return []

    snaps = await snapshot_fn(targets)
    reconciled: list[tuple[str, str]] = []
    for s in snaps:
        mid = s.get("name")
        dead = (not s.get("healthy")) and s.get("error") in DEAD_ERRORS
        if not dead:
            fails.pop(mid, None)  # 恢复 / 仅繁忙 → 清零
            continue
        fails[mid] = fails.get(mid, 0) + 1
        logger.warning(
            "vllm watchdog: %s 端口不可达(%s)%d/%d", mid, s.get("error"), fails[mid], fail_threshold)
        if fails[mid] < fail_threshold:
            continue  # 防抖:跨过 startup/restart 瞬时窗口
        try:
            action = await reconcile_fn(model_manager, mid)
            logger.warning("vllm watchdog: 自愈死掉实例 %s → %s", mid, action)
            reconciled.append((mid, action))
            if notify is not None:
                await notify(mid, action)
        except Exception as e:  # noqa: BLE001 —— 单个 reconcile 失败不拖垮巡检
            logger.error("vllm watchdog: reconcile %s 失败: %s", mid, e)
        finally:
            fails.pop(mid, None)  # 无论成败都清零;仍死会在后续轮重新累计后重试
    return reconciled


async def vllm_health_watchdog(
    model_manager,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
    notify: Callable[[str, str], Awaitable[None]] | None = None,
) -> None:
    """常驻后台 loop:每 `interval_s` 秒巡检一次,自愈死掉的 vLLM 实例。"""
    fails: dict[str, int] = {}
    while True:
        await asyncio.sleep(interval_s)
        try:
            await run_cycle(
                model_manager, fails, fail_threshold=fail_threshold, notify=notify)
        except Exception as e:  # noqa: BLE001 —— loop 永不因单轮异常退出
            logger.warning("vllm watchdog loop 异常: %s", e)
