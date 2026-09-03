"""开机加载策略:只有 resident 模型被预加载。

回归 2026-09-03 的坑:用户没给 `qwen3_6_35b_a3b_fp8` 开常驻(UI 菜单还显示「设为
自动加载」),开机它照样被拉到 GPU 1 占 34.9GB —— 因为 lifespan 有一条隐藏的
`_load_wf_deps`,把**所有 published 工作流**引用到的模型全 `load_model`,完全不看
resident 标记。工作流执行时 runner 本来就走 `get_or_load` 按需加载,预热非必需。

本文件锁两条不变式:
  1. lifespan 跑完后,published 工作流引用的非常驻模型**没有**被 load_model;
     引用登记(add_reference,防 idle/LRU 卸)照常发生。
  2. resident 模型仍旧走 preload_residents 预加载。

起 lifespan 的姿势抄 test_lane_k_lifespan_wiring.py(直接进 @asynccontextmanager)。
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import pytest

from src.models.service_instance import ServiceInstance
from src.models.workflow import Workflow


@pytest.fixture
def temp_pg_db(monkeypatch):
    """让 lifespan 跑在 conftest 建的临时 PG 库上(同 test_lane_k_lifespan_wiring)。"""
    from src.config import get_settings
    get_settings.cache_clear()
    import src.models.database as _db_mod
    monkeypatch.setattr(_db_mod, "_session_factory", None, raising=False)
    yield os.environ["DATABASE_URL"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_lifespan_does_not_load_models_of_published_workflows(
    monkeypatch, db_session, temp_pg_db
):
    """published 工作流引用了一个非常驻模型 → lifespan 只登记引用,绝不加载它。

    **必须开着后台任务跑**:被删的 `_load_wf_deps` 是一条后台 task,
    `NOUS_DISABLE_BG_TASKS=1`(conftest 默认)下它压根不起 —— 那样测的就不是回归了。
    所以这里把 gate 打开,同时把会碰 GPU / 动磁盘的两类 loop 换成 no-op:
    memory_guard / gpu_thermal_guard 会 poll nvidia-smi(CLAUDE.md:测试绝不碰 GPU),
    image orphan reaper 会真删产物目录里的旧文件。其余 loop 首步就 sleep,无害。
    """
    monkeypatch.setenv("NOUS_DISABLE_BG_TASKS", "0")
    monkeypatch.setenv("NOUS_DISABLE_VLLM_WATCHDOG", "1")
    monkeypatch.setenv("NOUS_DISABLE_STATUS_SAMPLER", "1")

    async def _noop_loop(*a, **kw):
        await asyncio.sleep(3600)

    import src.api.main as _main_mod
    import src.services.gpu_thermal_guard as _thermal_mod
    import src.services.image_output_storage as _img_store
    monkeypatch.setattr(_main_mod, "memory_guard_loop", _noop_loop)
    monkeypatch.setattr(_thermal_mod, "gpu_thermal_guard_loop", _noop_loop)
    monkeypatch.setattr(_img_store, "reap_orphans", lambda **kw: 0)

    wf = Workflow(name="新工作流", status="published", nodes=[], edges=[])
    db_session.add(wf)
    await db_session.commit()
    await db_session.refresh(wf)
    db_session.add(ServiceInstance(
        source_type="workflow", name="ltx-drama", type="inference",
        category="app", meter_dim="calls", workflow_id=wf.id,
    ))
    await db_session.commit()

    from src.services.model_manager import ModelManager

    loaded: list[str] = []
    refs: list[tuple[str, str]] = []

    async def _spy_load_model(self, model_id, *a, **kw):
        loaded.append(model_id)
        return None

    monkeypatch.setattr(
        ModelManager, "get_model_dependencies",
        lambda self, graph: [{"key": "qwen3_6_35b_a3b_fp8", "type": "llm"}],
    )
    monkeypatch.setattr(
        ModelManager, "add_reference",
        lambda self, key, ref: refs.append((key, ref)),
    )
    monkeypatch.setattr(ModelManager, "load_model", _spy_load_model)
    # resident preload 不是本用例的被测对象,换成 no-op(否则会照 models.yaml 真加载)。
    async def _noop_preload(self, on_loaded=None):
        return None
    monkeypatch.setattr(ModelManager, "preload_residents", _noop_preload)

    from src.api.main import create_app, lifespan as _lifespan

    app = create_app()
    async with _lifespan(app):
        # 让刚 create_task 的后台任务各跑第一步 —— 旧的 _load_wf_deps 起手无 sleep,
        # 第一次被调度就会 load_model。不 yield 的话它一步没跑就被 finally cancel 掉,
        # 本用例会假绿。
        for _ in range(5):
            await asyncio.sleep(0)

    # 引用登记照常(挡 idle checker / LRU 卸载)……
    assert (("qwen3_6_35b_a3b_fp8", str(wf.id))) in refs, \
        "published 工作流的模型引用仍应被登记(防 idle/LRU 卸载)"
    # ……但绝不因此加载模型。
    assert "qwen3_6_35b_a3b_fp8" not in loaded, \
        "非常驻模型不该因为被 published 工作流引用就开机加载(UI 只承诺常驻自动加载)"


@pytest.mark.asyncio
async def test_background_tasks_preload_residents_then_autostart(monkeypatch):
    """开机预加载只有一条后台任务,顺序做两件事:resident → autostart 服务。

    其余后台 loop(memory_guard 会 poll nvidia-smi 等)create_task 之后立刻 cancel,
    任务体一步都不跑(CLAUDE.md:测试绝不碰 GPU)。
    """
    monkeypatch.setenv("NOUS_DISABLE_BG_TASKS", "0")
    monkeypatch.setenv("NOUS_DISABLE_VLLM_WATCHDOG", "1")
    monkeypatch.setenv("NOUS_DISABLE_STATUS_SAMPLER", "1")

    order: list[str] = []

    from src.api.main import _start_background_tasks
    import src.services.service_autostart as _autostart_mod

    async def _fake_autostart(session_factory, mm, on_loaded=None):
        order.append("autostart")
        return []

    monkeypatch.setattr(_autostart_mod, "preload_autostart_services", _fake_autostart)

    model_mgr = MagicMock()

    async def _fake_preload(on_loaded=None):
        order.append("resident")

    model_mgr.preload_residents = MagicMock(side_effect=_fake_preload)
    app = MagicMock()
    app.state = MagicMock()

    bg, cache_t, resp_t, partial = _start_background_tasks(app, model_mgr)
    tasks = [t for t in (*bg, cache_t, resp_t, partial) if t is not None]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    # 预加载任务是唯一一条要跑完的 —— await 它,验两步的顺序。
    await app.state._resident_preload_task

    assert order == ["resident", "autostart"], \
        "开机预加载必须是 resident → autostart 的单任务顺序执行(别开第二个并发加载源)"
    # 后台没人绕过这两步直接 load_model。
    model_mgr.load_model.assert_not_called()
