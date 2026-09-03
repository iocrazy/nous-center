"""服务级「开机启动」:迁移、toggle API + 预览、开机只预加载 autostart 服务的模型。

语义(全仓一句话):开机只加载 ① resident 模型 ② autostart=true 服务引用到的模型,
其余等工作流执行时 runner 走 get_or_load 按需加载。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

# fresh_pg_db / _run_alembic 复用 alembic baseline 测试那套(全新空库 + 真 alembic CLI)。
from tests.test_alembic_baseline import (  # noqa: F401 — fresh_pg_db 是 fixture,靠 import 注入
    _pg_fetch,
    _run_alembic,
    fresh_pg_db,
)
from tests.conftest import _mock_model_manager, _test_engine

from src.models.service_instance import ServiceInstance


def _llm_snapshot(model_key: str) -> dict:
    """发布快照的最小形状(见 workflow_publish._build_snapshot)。"""
    return {
        "schema": "comfy/api-1",
        "nodes": {"n1": {"class_type": "llm", "inputs": {"model_key": model_key}}},
    }


def _svc(name: str, **over) -> ServiceInstance:
    kwargs = dict(
        source_type="workflow",
        name=name,
        type="inference",
        category="app",
        meter_dim="calls",
        status="active",
        workflow_snapshot={},
    )
    kwargs.update(over)
    return ServiceInstance(**kwargs)


class _FakeRegistry:
    """只需要 .get(key) —— 真 ModelRegistry 的最小契约。"""

    def __init__(self, specs: dict):
        self._specs = specs

    def get(self, key):
        return self._specs.get(key)


def _spec(mid, vram_mb=16384, gpu=1, resident=False):
    return SimpleNamespace(id=mid, vram_mb=vram_mb, gpu=gpu, resident=resident)


def _session_factory():
    return async_sessionmaker(_test_engine(), expire_on_commit=False)


@pytest.fixture
async def svc_client():
    """跟 conftest 的 db_client 同构,但把 app 也 yield 出来 —— 要往 app.state 上塞
    registry(预览里的 vram/gpu 从它取)。"""
    from src.api.main import create_app
    from src.models.database import get_async_session

    engine = _test_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.state.model_manager = _mock_model_manager()
    app.dependency_overrides[get_async_session] = override_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as c:
        yield c, app

    await engine.dispose()


# ---------------------------------------------------------------- 迁移


def test_upgrade_head_adds_service_instances_autostart(fresh_pg_db):  # noqa: F811 — 导入的 fixture
    """全新库 `alembic upgrade head` 后必须有 service_instances.autostart。

    NOT NULL + server_default false —— 存量行加列不能炸,默认必须是「不开机启动」。
    """
    db = fresh_pg_db
    up = _run_alembic(["upgrade", "head"], db)
    assert up.returncode == 0, f"upgrade head failed:\n{up.stdout}\n{up.stderr}"

    rows = _pg_fetch(
        db.replace("postgresql+asyncpg://", "postgresql://"),
        "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_name = 'service_instances' AND column_name = 'autostart'",
    )
    assert rows, "upgrade head 后 service_instances 没有 autostart 列"
    assert rows[0]["data_type"] == "boolean", rows[0]
    assert rows[0]["is_nullable"] == "NO", rows[0]
    assert "false" in (rows[0]["column_default"] or ""), rows[0]


# ---------------------------------------------------------------- API


@pytest.mark.asyncio
async def test_autostart_toggle_roundtrip_and_preview(svc_client, db_session):
    client, app = svc_client
    svc = _svc("ltx-drama", workflow_snapshot=_llm_snapshot("qwen3_6_35b_a3b_fp8"))
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)

    app.state.model_manager._registry = _FakeRegistry(
        {"qwen3_6_35b_a3b_fp8": _spec("qwen3_6_35b_a3b_fp8", vram_mb=35840, gpu=1)}
    )

    # 默认关。
    r = await client.get(f"/api/v1/services/{svc.id}")
    assert r.status_code == 200, r.text
    assert r.json()["autostart"] is False

    # 预览:开机会加载什么(只读,不改状态)。
    r = await client.get(f"/api/v1/services/{svc.id}/autostart-preview")
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["autostart"] is False
    assert preview["preload_models"] == [
        {"name": "qwen3_6_35b_a3b_fp8", "vram_gb": 35.0, "gpu": 1}
    ]

    # 开。
    r = await client.post(f"/api/v1/services/{svc.id}/autostart", json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["autostart"] is True
    assert [m["name"] for m in body["preload_models"]] == ["qwen3_6_35b_a3b_fp8"]

    # 列表也带上了(写路径 invalidate 过 services 缓存,读到的是新值)。
    r = await client.get("/api/v1/services")
    assert [s["autostart"] for s in r.json() if s["name"] == "ltx-drama"] == [True]

    # 关回去。
    r = await client.post(f"/api/v1/services/{svc.id}/autostart", json={"enabled": False})
    assert r.status_code == 200, r.text
    assert r.json()["autostart"] is False
    await db_session.refresh(svc)
    assert svc.autostart is False


@pytest.mark.asyncio
async def test_autostart_endpoints_404_on_missing_service(svc_client):
    client, _ = svc_client
    assert (await client.get("/api/v1/services/999999/autostart-preview")).status_code == 404
    r = await client.post("/api/v1/services/999999/autostart", json={"enabled": True})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_preview_lists_nothing_for_component_only_service(svc_client, db_session):
    """图像服务引用的是组件文件,开机不预加载 → 预览必须诚实地为空(不许乱承诺)。"""
    client, _ = svc_client
    svc = _svc("img-app", workflow_snapshot={
        "nodes": {"n1": {"class_type": "flux2_load_diffusion_model",
                         "inputs": {"file": "/models/flux2.safetensors"}}},
    })
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)

    r = await client.get(f"/api/v1/services/{svc.id}/autostart-preview")
    assert r.status_code == 200, r.text
    assert r.json()["preload_models"] == []


@pytest.mark.asyncio
async def test_preview_for_model_sourced_service(svc_client, db_session):
    """source_type=model 的服务(asr/embedding 等)绑的就是它自己那个引擎。"""
    client, app = svc_client
    svc = _svc("qwen3-asr", source_type="model", source_name="moss-asr", category="asr")
    db_session.add(svc)
    await db_session.commit()
    await db_session.refresh(svc)
    app.state.model_manager._registry = _FakeRegistry(
        {"moss-asr": _spec("moss-asr", vram_mb=8192, gpu=2)})

    r = await client.get(f"/api/v1/services/{svc.id}/autostart-preview")
    assert r.json()["preload_models"] == [
        {"name": "moss-asr", "vram_gb": 8.0, "gpu": 2}
    ]


# ---------------------------------------------------------------- 开机预加载


def _mm(registry, loaded: set[str] | None = None):
    loaded = loaded or set()
    mm = MagicMock()
    mm._registry = registry
    mm.is_loaded = MagicMock(side_effect=lambda k: k in loaded)
    calls: list[str] = []

    async def _load(key, *a, **kw):
        calls.append(key)

    mm.load_model = MagicMock(side_effect=_load)
    mm.calls = calls
    return mm


@pytest.mark.asyncio
async def test_preload_only_autostart_services(db_session):
    """只加载 autostart=true 服务的模型;没开的服务一个都不碰。"""
    from src.services.service_autostart import preload_autostart_services

    db_session.add_all([
        _svc("on-svc", workflow_snapshot=_llm_snapshot("model-on"), autostart=True),
        _svc("off-svc", workflow_snapshot=_llm_snapshot("model-off")),
    ])
    await db_session.commit()

    mm = _mm(_FakeRegistry({"model-on": _spec("model-on"), "model-off": _spec("model-off")}))
    loaded = await preload_autostart_services(_session_factory(), mm)

    assert loaded == ["model-on"]
    assert mm.calls == ["model-on"], "没开开机启动的服务,它的模型绝不能被加载"


@pytest.mark.asyncio
async def test_preload_skips_resident_loaded_and_inactive(db_session):
    """跳过:resident(preload_residents 管了)、已加载的、非 active 服务。"""
    from src.services.service_autostart import preload_autostart_services

    db_session.add_all([
        _svc("a", workflow_snapshot=_llm_snapshot("m-resident"), autostart=True),
        _svc("b", workflow_snapshot=_llm_snapshot("m-loaded"), autostart=True),
        _svc("c", workflow_snapshot=_llm_snapshot("m-fresh"), autostart=True),
        _svc("d", workflow_snapshot=_llm_snapshot("m-retired"),
             autostart=True, status="retired"),
    ])
    await db_session.commit()

    mm = _mm(
        _FakeRegistry({
            "m-resident": _spec("m-resident", resident=True),
            "m-loaded": _spec("m-loaded"),
            "m-fresh": _spec("m-fresh"),
            "m-retired": _spec("m-retired"),
        }),
        loaded={"m-loaded"},
    )
    loaded = await preload_autostart_services(_session_factory(), mm)

    assert loaded == ["m-fresh"]
    assert mm.calls == ["m-fresh"]


@pytest.mark.asyncio
async def test_preload_is_fail_soft(db_session):
    """一个模型加载失败不许影响后面的(同 preload_residents 的 fail-soft 约定)。"""
    from src.services.service_autostart import preload_autostart_services

    db_session.add(_svc("multi", autostart=True, workflow_snapshot={
        "nodes": {
            "n1": {"class_type": "llm", "inputs": {"model_key": "boom"}},
            "n2": {"class_type": "tts_engine", "inputs": {"engine": "ok"}},
        },
    }))
    await db_session.commit()

    mm = _mm(_FakeRegistry({"boom": _spec("boom"), "ok": _spec("ok")}))
    calls: list[str] = []

    async def _load(key, *a, **kw):
        calls.append(key)
        if key == "boom":
            raise RuntimeError("OOM")

    mm.load_model = MagicMock(side_effect=_load)
    loaded = await preload_autostart_services(_session_factory(), mm)

    assert calls == ["boom", "ok"], "第一个失败不该中断后面的"
    assert loaded == ["ok"]


@pytest.mark.asyncio
async def test_preload_noop_without_autostart_services(db_session):
    from src.services.service_autostart import preload_autostart_services

    db_session.add(_svc("plain", workflow_snapshot=_llm_snapshot("m")))
    await db_session.commit()

    mm = _mm(_FakeRegistry({"m": _spec("m")}))
    assert await preload_autostart_services(_session_factory(), mm) == []
    assert mm.calls == []
