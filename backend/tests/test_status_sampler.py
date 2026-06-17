"""status 页采样器 + 聚合单测 —— CI 可跑(SQLite,无 GPU/真 vLLM)。

覆盖:① worst() ② compute_statuses 各组件判定(DB/llm/embedding/runner/gpu,含 vLLM
不健康→down、runner 缺失→down、无实例→operational)③ uptime_history 按天分桶 + 总 uptime%
+ 无数据天标 nodata ④ sample_once 落库。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.status_sample import StatusSample
from src.services import status_sampler as ss


@pytest.fixture
async def session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/s.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as s:
        yield s
    await engine.dispose()


def _entry(loaded=True, port=40000, mtype="llm"):
    return SimpleNamespace(
        adapter=SimpleNamespace(is_loaded=loaded, _port=port),
        spec=SimpleNamespace(model_type=mtype),
    )


def _sup(group, running=True, n_loaded=1):
    return SimpleNamespace(group_id=group, is_running=running,
                           loaded_models=[{"id": f"m{i}"} for i in range(n_loaded)])


def _app_state(models=None, runners=None):
    # runners: list of supervisor namespaces; 默认 image+tts 各加载 1 个模型(operational)
    sups = runners if runners is not None else [_sup("image"), _sup("tts")]
    return SimpleNamespace(
        model_manager=SimpleNamespace(_models=models or {}),
        runner_supervisors=sups,
    )


# ---------- worst ----------

def test_worst_ranking():
    assert ss.worst([]) == "operational"
    assert ss.worst(["operational", "operational"]) == "operational"
    assert ss.worst(["operational", "degraded"]) == "degraded"
    assert ss.worst(["degraded", "down", "operational"]) == "down"
    # idle 视同 operational —— 不把 overall 拉成异常。
    assert ss.worst(["operational", "idle"]) == "operational"
    assert ss.worst(["idle", "down"]) == "down"


# ---------- compute_statuses ----------

@pytest.mark.asyncio
async def test_compute_all_healthy(session, monkeypatch):
    async def snap(targets): return [{"name": m, "healthy": True} for m, _ in targets]
    monkeypatch.setattr("src.services.vllm_metrics.snapshot_all", snap)
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1, 2, 3])
    st = _app_state(models={"q": _entry(mtype="llm"), "e": _entry(port=40001, mtype="embedding")})
    out = await ss.compute_statuses(st, session)
    assert out == {"backend": "operational", "database": "operational", "llm": "operational",
                   "embedding": "operational", "image": "operational", "tts": "operational",
                   "gpu": "operational"}


@pytest.mark.asyncio
async def test_compute_llm_down_when_vllm_unhealthy(session, monkeypatch):
    async def snap(targets): return [{"name": m, "healthy": False, "error": "ConnectError"} for m, _ in targets]
    monkeypatch.setattr("src.services.vllm_metrics.snapshot_all", snap)
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    st = _app_state(models={"q": _entry(mtype="llm")})
    out = await ss.compute_statuses(st, session)
    assert out["llm"] == "down"
    assert out["embedding"] == "idle"   # 无 embedding 实例 → idle(没装,不是 operational)


@pytest.mark.asyncio
async def test_compute_vllm_idle_when_no_instances(session, monkeypatch):
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    out = await ss.compute_statuses(_app_state(models={}), session)
    assert out["llm"] == "idle" and out["embedding"] == "idle"


@pytest.mark.asyncio
async def test_compute_runner_idle_vs_serving_vs_down(session, monkeypatch):
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    st = _app_state(models={}, runners=[
        _sup("image", running=True, n_loaded=2),   # 有模型 → operational
        _sup("tts", running=True, n_loaded=0),      # 进程活但没装 → idle(用户反馈点)
    ])
    out = await ss.compute_statuses(st, session)
    assert out["image"] == "operational"
    assert out["tts"] == "idle"


@pytest.mark.asyncio
async def test_compute_runner_missing_is_down(session, monkeypatch):
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    st = _app_state(models={}, runners=[_sup("image", n_loaded=1)])  # 缺 tts
    out = await ss.compute_statuses(st, session)
    assert out["image"] == "operational"
    assert out["tts"] == "down"


@pytest.mark.asyncio
async def test_compute_gpu_down_when_none(session, monkeypatch):
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [])
    out = await ss.compute_statuses(_app_state(), session)
    assert out["gpu"] == "down"


# ---------- sample_once + uptime_history ----------

@pytest.mark.asyncio
async def test_sample_once_writes_all_components(session, monkeypatch):
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    await ss.sample_once(_app_state(), session)
    from sqlalchemy import select, func
    n = (await session.execute(select(func.count()).select_from(StatusSample))).scalar()
    assert n == len(ss.COMPONENT_KEYS)


@pytest.mark.asyncio
async def test_uptime_counts_idle_as_up(session):
    """idle(在线没装模型)算可用 —— 否则没常驻模型的组件 uptime 永远 ~0% 误报。"""
    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    session.add_all([
        StatusSample(component="tts", status="idle", ts=now),
        StatusSample(component="tts", status="idle", ts=now),
        StatusSample(component="tts", status="operational", ts=now),
        StatusSample(component="tts", status="down", ts=now),
    ])
    await session.commit()
    hist = await ss.uptime_history(session, days=7)
    # 3 up(2 idle + 1 op)/ 4 = 75%(若 idle 不算 up 会是 25%)
    assert hist["tts"]["uptime_pct"] == pytest.approx(75.0, abs=0.01)


@pytest.mark.asyncio
async def test_uptime_history_buckets_and_pct(session):
    now = datetime.now(timezone.utc)
    today = now.replace(hour=12, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    # database: 今天 3 op + 1 down = 75%;昨天 2 op = 100%
    session.add_all([
        StatusSample(component="database", status="operational", ts=today),
        StatusSample(component="database", status="operational", ts=today),
        StatusSample(component="database", status="operational", ts=today),
        StatusSample(component="database", status="down", ts=today),
        StatusSample(component="database", status="operational", ts=yesterday),
        StatusSample(component="database", status="operational", ts=yesterday),
    ])
    await session.commit()

    hist = await ss.uptime_history(session, days=7)
    db = hist["database"]
    assert len(db["days"]) == 7
    # 总:5 op / 6 = 83.33
    assert db["uptime_pct"] == pytest.approx(83.33, abs=0.01)
    today_bucket = db["days"][-1]
    assert today_bucket["uptime_pct"] == pytest.approx(75.0, abs=0.01)
    assert today_bucket["status"] == "degraded"   # 75% → degraded
    # 没采样的组件 → 每天 nodata
    assert all(d["status"] == "nodata" for d in hist["gpu"]["days"])
    assert hist["gpu"]["uptime_pct"] is None


# ---------- 端点(admin 在测试里关闭)----------

@pytest.mark.asyncio
async def test_status_endpoint_shape(app, client, tmp_path, monkeypatch):
    app.state.model_manager = SimpleNamespace(_models={})
    app.state.runner_supervisors = []
    monkeypatch.setattr("src.services.gpu_monitor.get_gpu_stats", lambda: [1])
    # 用 SQLite override DB 依赖,避免本地跑碰生产 PG(uv 不 load .env)。
    from src.models.database import get_async_session
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ep.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    async def _override():
        async with sf() as s:
            yield s
    app.dependency_overrides[get_async_session] = _override
    try:
        r = await client.get("/api/v1/status")
    finally:
        app.dependency_overrides.pop(get_async_session, None)
        await engine.dispose()
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"overall", "updated_at", "components"}
    keys = [c["key"] for c in body["components"]]
    assert keys == ss.COMPONENT_KEYS
    for c in body["components"]:
        assert set(c) >= {"key", "name", "status", "uptime_7d", "days"}
        assert len(c["days"]) == 7
