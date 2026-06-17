"""状态页采样器 + 聚合(2026-06-17,status 页 v1)。

- compute_statuses():现算各组件当前状态(端点实时用,不读 DB)。
- status_sampler_loop():后台每 60s 把当前状态落 status_samples,定期清 8 天外旧行。
- uptime_history():从 status_samples 聚合「过去 N 天每组件每天 uptime%」给 status 页画条。

组件故意**不含** Lane-K llm runner supervisor —— 它常驻 running:false 但 vLLM 由
model_manager 独立 spawn、服务正常,纳入会恒 degraded(噪声,2026-06-16 巡检脚本同理)。
LLM 组件 = 直接看 llm 型 vLLM 实例健康。
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.status_sample import StatusSample

logger = logging.getLogger(__name__)

OPERATIONAL = "operational"
DEGRADED = "degraded"
DOWN = "down"
_RANK = {OPERATIONAL: 0, DEGRADED: 1, DOWN: 2}

# (key, 显示名)。顺序即页面展示顺序。
COMPONENTS: list[tuple[str, str]] = [
    ("backend", "后端 API"),
    ("database", "数据库"),
    ("llm", "LLM 推理 (vLLM)"),
    ("embedding", "向量 (vLLM)"),
    ("image", "图像 Runner"),
    ("tts", "语音 Runner"),
    ("gpu", "GPU"),
]
COMPONENT_KEYS = [k for k, _ in COMPONENTS]

DEFAULT_INTERVAL_S = 60.0
RETENTION_DAYS = 8
PRUNE_EVERY_S = 3600.0


def worst(statuses) -> str:
    """多个状态取最差(down > degraded > operational)。空 → operational。"""
    s = list(statuses)
    if not s:
        return OPERATIONAL
    return max(s, key=lambda x: _RANK.get(x, 0))


def _vllm_targets_by_type(model_manager):
    """已加载且有端口的 vLLM 实例,按 model_type 分组 → {type: [(mid, port)]}。"""
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for mid, entry in getattr(model_manager, "_models", {}).items():
        adapter = getattr(entry, "adapter", None)
        if not getattr(adapter, "is_loaded", False):
            continue
        port = getattr(adapter, "_port", None) or getattr(adapter, "port", None)
        if not port:
            continue
        mtype = getattr(getattr(entry, "spec", None), "model_type", None) or "llm"
        out[str(mtype)].append((mid, int(port)))
    return out


async def _vllm_component_status(targets) -> str:
    """一组 vLLM 实例(同类)的组件状态:无实例=operational(没东西可挂);全健康=
    operational;任一连不上=down。"""
    if not targets:
        return OPERATIONAL
    from src.services.vllm_metrics import snapshot_all
    snaps = await snapshot_all(targets)
    if any(not s.get("healthy") for s in snaps):
        return DOWN
    return OPERATIONAL


async def compute_statuses(app_state, session: AsyncSession | None = None) -> dict[str, str]:
    """现算每个组件当前状态。每项独立 try —— 单项探测失败记 down,绝不让采样/端点崩。"""
    out: dict[str, str] = {}

    # backend:能跑到这就是活的。
    out["backend"] = OPERATIONAL

    # database:SELECT 1。
    try:
        if session is not None:
            await session.execute(text("SELECT 1"))
            out["database"] = OPERATIONAL
        else:
            from src.models.database import get_session_factory
            async with get_session_factory()() as s:
                await s.execute(text("SELECT 1"))
            out["database"] = OPERATIONAL
    except Exception as e:  # noqa: BLE001
        logger.warning("status: database check failed: %s", e)
        out["database"] = DOWN

    mgr = getattr(app_state, "model_manager", None)
    by_type = _vllm_targets_by_type(mgr) if mgr else {}

    # llm / embedding:对应 type 的 vLLM 实例健康。
    try:
        out["llm"] = await _vllm_component_status(by_type.get("llm", []))
    except Exception as e:  # noqa: BLE001
        logger.warning("status: llm check failed: %s", e)
        out["llm"] = DOWN
    try:
        out["embedding"] = await _vllm_component_status(by_type.get("embedding", []))
    except Exception as e:  # noqa: BLE001
        logger.warning("status: embedding check failed: %s", e)
        out["embedding"] = DOWN

    # image / tts runner:对应 supervisor running。
    runners = {}
    for sup in getattr(app_state, "runner_supervisors", []) or []:
        try:
            snap = sup.health_snapshot()
            runners[snap.get("group_id")] = bool(snap.get("running"))
        except Exception:  # noqa: BLE001
            pass
    for key in ("image", "tts"):
        if key in runners:
            out[key] = OPERATIONAL if runners[key] else DOWN
        else:
            out[key] = DOWN  # 应有的 runner 不在 → 异常

    # gpu:nvidia-smi 能列出卡。
    try:
        from src.services.gpu_monitor import get_gpu_stats
        out["gpu"] = OPERATIONAL if len(get_gpu_stats()) > 0 else DOWN
    except Exception as e:  # noqa: BLE001
        logger.warning("status: gpu check failed: %s", e)
        out["gpu"] = DOWN

    return out


async def sample_once(app_state, session: AsyncSession) -> None:
    statuses = await compute_statuses(app_state, session)
    now = datetime.now(timezone.utc)
    session.add_all([
        StatusSample(component=k, status=v, ts=now) for k, v in statuses.items()
    ])
    await session.commit()


async def _prune(session: AsyncSession) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    await session.execute(delete(StatusSample).where(StatusSample.ts < cutoff))
    await session.commit()


async def status_sampler_loop(app_state, *, interval_s: float = DEFAULT_INTERVAL_S) -> None:
    """后台:每 interval_s 采样一次;每 PRUNE_EVERY_S 清一次旧行。loop 永不因单轮异常退出。"""
    from src.models.database import get_session_factory
    last_prune = 0.0
    elapsed = 0.0
    while True:
        await asyncio.sleep(interval_s)
        elapsed += interval_s
        try:
            async with get_session_factory()() as session:
                await sample_once(app_state, session)
                if elapsed - last_prune >= PRUNE_EVERY_S:
                    await _prune(session)
                    last_prune = elapsed
        except Exception as e:  # noqa: BLE001
            logger.warning("status sampler loop error: %s", e)


async def uptime_history(session: AsyncSession, *, days: int = 7) -> dict[str, dict]:
    """聚合每组件「过去 days 天每天 uptime%」+ 总 uptime%。

    返回 {component: {"uptime_pct": float, "days": [{"date","uptime_pct","status","samples"}]}}。
    用 func.date(ts) 按 UTC 日分桶(PG/SQLite 都支持),桶内 uptime=operational 占比。
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    day_col = func.date(StatusSample.ts)
    ok_col = func.sum(case((StatusSample.status == OPERATIONAL, 1), else_=0))
    rows = (await session.execute(
        select(StatusSample.component, day_col.label("d"),
               func.count().label("total"), ok_col.label("ok"))
        .where(StatusSample.ts >= since)
        .group_by(StatusSample.component, day_col)
    )).all()

    # {component: {date_str: (ok, total)}}
    acc: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for comp, d, total, ok in rows:
        acc[comp][str(d)] = (int(ok or 0), int(total or 0))

    today = datetime.now(timezone.utc).date()
    date_keys = [str(today - timedelta(days=days - 1 - i)) for i in range(days)]

    result: dict[str, dict] = {}
    for comp in COMPONENT_KEYS:
        per_day = acc.get(comp, {})
        day_list = []
        tot_ok = tot_all = 0
        for dk in date_keys:
            ok, total = per_day.get(dk, (0, 0))
            tot_ok += ok
            tot_all += total
            if total == 0:
                day_list.append({"date": dk, "uptime_pct": None, "status": "nodata", "samples": 0})
            else:
                pct = round(100.0 * ok / total, 2)
                # 当天有任一非 operational 即标 degraded/down(画条用):全绿 operational,
                # 否则按最差。简单用 pct 阈值:100=operational,>0=degraded,0=down。
                st = OPERATIONAL if pct >= 100 else (DOWN if pct == 0 else DEGRADED)
                day_list.append({"date": dk, "uptime_pct": pct, "status": st, "samples": total})
        overall = round(100.0 * tot_ok / tot_all, 2) if tot_all else None
        result[comp] = {"uptime_pct": overall, "days": day_list}
    return result
