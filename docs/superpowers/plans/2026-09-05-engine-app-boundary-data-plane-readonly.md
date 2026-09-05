# nous-engine 数据面对模型放置只读 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `/v1/*` 推理端点永远不能改变显存里有什么:未就绪即刻 503、`/v1/models` 只列就绪、`unload` 说真话、常驻只由 `resident: true` 决定。

**Architecture:** 数据面(`/v1/*` 五个兼容路由)只经只读的 `get_vllm_base_url` 解析端点;唯一的懒加载入口 `ensure_vllm_base_url` 删除,并用静态守卫测试锁住。已发布工作流的进程级模型引用整体废除,`resident` 成为唯一钉住手段;`unload` 路由接住 `unload_model` 的返回值。每个 Task 一个独立 PR,顺序不可调换(后面的 Task 依赖前面的 helper)。

**Tech Stack:** FastAPI + SQLAlchemy async(PostgreSQL)、pytest(`-n 8`,PostgreSQL 临时库)、vLLM 子进程(测试里 mock `subprocess.Popen`,conftest 有护栏)。

**Spec:** `docs/superpowers/specs/2026-09-05-engine-app-boundary-data-plane-readonly-design.md`

## Global Constraints

- 测试只跑 PostgreSQL;`backend/tests/conftest.py` 会按 `.env` 的 `DATABASE_URL` 建临时库。跑法:`cd backend && uv run pytest <files> -n 8 -q`。
- **绝不**跑会起真推理服务的用例;conftest 的 Popen 护栏会把 `vllm.entrypoints` 直接 AssertionError,本计划不涉及 GPU。
- **绝不** `uv sync`(生产 venv,裸 sync 裁掉 vllm/torch)。
- 每个 Task:从 `master` 开分支 → 提交 → `git push -u origin <branch>` → `gh pr create --base master` → CI 绿后 squash merge → 工作树切回 `master` 并 `git pull --ff-only`。生产从这棵工作树服务,**不要把工作树留在功能分支上**。
- 提交用 `git -c commit.gpgsign=false commit`(本机无签名钥)。
- 错误信封统一走 `src/errors.py` 的 `NousError` 子类(`{"error": {message,type,code,param,fix}}`),不再新增 `HTTPException(503, detail=str)`。
- 服务名 ↔ 引擎名映射的**唯一写法**:`instance.source_name or str(instance.source_id)`。
- 所有改动的中文注释风格跟随文件既有注释(记「为什么」,带日期)。

---

### Task 1: `unload` 诚实化 + `held_by` 可观测(PR-1)

**Files:**
- Modify: `backend/src/api/routes/engines.py:411-428`(`unload_engine`)、`:97-165`(`_build_engine_info`)
- Modify: `backend/src/models/schemas.py:80-130`(`EngineInfo` 加 `held_by`)
- Modify: `backend/src/services/model_manager.py:836-862`(`unload_model` 日志级别)
- Modify: `backend/tests/conftest.py:309-320`(`_mock_model_manager` 补两个默认)
- Test: `backend/tests/test_api_engines.py`

**Interfaces:**
- Consumes: `ModelManager.unload_model(model_id, force=False) -> bool`(拒绝返回 `False`)、`ModelManager.get_references(model_id) -> set[str]`、`ModelManager.is_loaded(model_id) -> bool`、`src.errors.ConflictError`。
- Produces: `EngineInfo.held_by: list[str]`(GET `/api/v1/engines` 每项);`POST /api/v1/engines/{name}/unload` 被拒 → 409,`error.code ∈ {"engine_referenced", "engine_in_use"}`。

- [ ] **Step 1: conftest 的 mock manager 补默认值**

`backend/tests/conftest.py` 的 `_mock_model_manager()` 在 `mgr.check_idle_models = ...` 之后加两行(现有用例不受影响:默认「没加载、没引用」):

```python
    mgr.is_loaded = MagicMock(return_value=False)
    mgr.get_references = MagicMock(return_value=set())
```

- [ ] **Step 2: 写失败测试(三条)**

追加到 `backend/tests/test_api_engines.py` 末尾:

```python
async def test_unload_referenced_engine_returns_409_with_reason(client):
    """spec §8:unload_model 返回 False 且模型仍 loaded → 409 engine_referenced,不再假报 unloaded。
    2026-09-05 实测:有引用时路由返回 200 'unloaded',进程活着、20.5G 显存不退。"""
    from unittest.mock import AsyncMock, MagicMock
    mgr = client._transport.app.state.model_manager
    mgr.unload_model = AsyncMock(return_value=False)
    mgr.is_loaded = MagicMock(return_value=True)
    mgr.get_references = MagicMock(return_value={"308084173191516160"})

    resp = await client.post("/api/v1/engines/qwen3_tts_base/unload")
    assert resp.status_code == 409, resp.text
    err = resp.json()["error"]
    assert err["code"] == "engine_referenced"
    assert "308084173191516160" in err["message"]
    assert "force=true" in err["fix"]


async def test_unload_in_use_engine_returns_409_and_force_does_not_help(client):
    from unittest.mock import AsyncMock, MagicMock
    mgr = client._transport.app.state.model_manager
    mgr.unload_model = AsyncMock(return_value=False)
    mgr.is_loaded = MagicMock(return_value=True)
    mgr.get_references = MagicMock(return_value=set())   # 没引用、非常驻 → 只能是 in_use

    resp = await client.post("/api/v1/engines/qwen3_tts_base/unload?force=true")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "engine_in_use"
    mgr.unload_model.assert_awaited_once_with("qwen3_tts_base", force=True)


async def test_engines_list_exposes_held_by(db_client):
    """spec §9:为什么它还在显存里,要能从 /api/v1/engines 一眼看出来。"""
    from unittest.mock import MagicMock
    mgr = db_client._transport.app.state.model_manager
    mgr.loaded_model_ids = ["qwen3_tts_base"]
    mgr.get_references = MagicMock(
        side_effect=lambda mid: {"proxy-abc"} if mid == "qwen3_tts_base" else set())

    resp = await db_client.get("/api/v1/engines")
    assert resp.status_code == 200
    by_name = {e["name"]: e for e in resp.json()}
    assert by_name["qwen3_tts_base"]["held_by"] == ["proxy-abc"]
    others = [e for n, e in by_name.items() if n != "qwen3_tts_base"]
    assert all(e["held_by"] == [] for e in others)
```

- [ ] **Step 3: 跑,确认失败**

Run: `cd backend && uv run pytest tests/test_api_engines.py -k "referenced or in_use or held_by" -q`
Expected: 3 failed —— 前两条 `assert 200 == 409`,第三条 `KeyError: 'held_by'`。

- [ ] **Step 4: `EngineInfo` 加字段**

`backend/src/models/schemas.py`,在 `loaded_gpus: list[int] | None = None` 之后加:

```python
    # spec 2026-09-05 §9:当前持有该模型的活跃引用(正常为空,或只剩请求期的 `proxy-*`)。
    # 「为什么它还在显存里」的答案只有两种:resident=True,或 held_by 非空。没有第三种。
    held_by: list[str] = []
```

- [ ] **Step 5: `_build_engine_info` 填 `held_by`**

`backend/src/api/routes/engines.py` 的 `info = EngineInfo(...)` 里,在 `loaded_gpus=...` 之后加一项:

```python
        held_by=sorted(request.app.state.model_manager.get_references(key))
        if (loaded and request is not None) else [],
```

- [ ] **Step 6: 重写 `unload_engine`**

替换 `backend/src/api/routes/engines.py:411-428` 整个函数为:

```python
@router.post("/{name}/unload", response_model=EngineLoadResponse, dependencies=[Depends(require_admin)])
async def unload_engine(name: str, request: Request, force: bool = False):
    configs = scan_models()
    if name not in configs:
        raise HTTPException(404, detail=f"Unknown engine: {name}")

    cfg = configs[name]
    if cfg.get("resident", False) and not force:
        raise HTTPException(409, detail=f"Engine {name} is resident. Use force=true to unload.")

    model_mgr = request.app.state.model_manager
    ok = await model_mgr.unload_model(name, force=force)
    # spec 2026-09-05 §8:接住返回值。此前这里无条件报 "unloaded",而 unload_model 在
    # 有引用 / in_use 时 return False 且只打 debug —— 真机表现是 200 + 进程活着 + 显存不退。
    # 「没加载」的 False 仍是 no-op 200(与 test_unload_non_loaded_engine 一致)。
    if ok is False and model_mgr.is_loaded(name):
        from src.errors import ConflictError  # noqa: PLC0415
        refs = sorted(model_mgr.get_references(name))
        if refs:
            raise ConflictError(
                f"Engine {name} is referenced by {refs}; not unloaded.",
                code="engine_referenced",
                fix=f"POST /api/v1/engines/{name}/unload?force=true",
            )
        raise ConflictError(
            f"Engine {name} is in use (inference in flight); not unloaded.",
            code="engine_in_use",
            fix="wait for in-flight requests to finish, then retry; force=true does not override in_use",
        )

    # round9 BUG4:清掉残留的 loading/failed 状态。_build_engine_info 里
    # _loading_states 优先级高于 loaded/unloaded —— load 失败写了 {"status":"failed"}
    # 后从不被清(unload 旧实现不 pop),GET /engines 会永远显示 "failed",哪怕重新
    # unload 也甩不掉。卸载即代表该 engine 回到干净 unloaded 态,这里 pop 掉。
    _loading_states.pop(name, None)

    invalidate("engines")
    return EngineLoadResponse(name=name, status="unloaded")
```

- [ ] **Step 7: `unload_model` 拒绝时日志提到 info**

`backend/src/services/model_manager.py` 的 `unload_model` 里两处 `logger.debug("Skipping unload of resident model %r", ...)` 与 `logger.debug("Skipping unload of referenced model %r (refs=%s)", ...)` 改为 `logger.info`。其余不动。

- [ ] **Step 8: 跑,确认通过 + 无回归**

Run: `cd backend && uv run pytest tests/test_api_engines.py tests/test_engines_list_runtime_override_freshness.py -n 8 -q`
Expected: 全部 PASS(含 `test_unload_non_loaded_engine` 仍 200)。

- [ ] **Step 9: 提交、PR**

```bash
git checkout -b fix/unload-honest-held-by master
git add backend/src/api/routes/engines.py backend/src/models/schemas.py backend/src/services/model_manager.py backend/tests/conftest.py backend/tests/test_api_engines.py
git -c commit.gpgsign=false commit -m "fix(engines): unload 被拒返回 409 而不是假报 unloaded;/api/v1/engines 暴露 held_by

spec 2026-09-05 §8/§9。此前 unload_engine 丢弃 unload_model 的返回值无条件报
'unloaded';有引用时真机表现是 200 + vLLM 进程活着 + 20.5G 显存不退。"
git push -u origin fix/unload-honest-held-by && gh pr create --base master --fill
```

CI 绿后 `gh pr merge <n> --squash --delete-branch && git checkout master && git pull --ff-only`。

---

### Task 2: 废除已发布工作流的长期模型引用(PR-2)

**Files:**
- Modify: `backend/src/services/startup_reconcile.py:14-50`
- Modify: `backend/src/api/routes/workflows.py:109-140`(`unpublish_workflow`)
- Modify: `backend/src/api/routes/services.py:622-647`(删服务时的工作流回退段)
- Test: `backend/tests/test_startup_reconcile.py`、`backend/tests/test_startup_model_load_policy.py`、新增 `backend/tests/test_idle_ttl_ignores_workflows.py`

**Interfaces:**
- Consumes: `ModelManager.check_idle_models()`(只看 `spec.resident` / `_references` / `spec.ttl_seconds` / `last_used`)。
- Produces: `reconcile_orphan_published_workflows(session, model_mgr) -> int` 不再调 `add_reference`;下架/删服务不再调 `remove_reference` / `unload_model`。`proxy_ref`(`openai_compat.py:352`)**保留**。

- [ ] **Step 1: 翻转两条既有断言(先让它们失败)**

`backend/tests/test_startup_reconcile.py` 第 43-45 行,把

```python
    mm.add_reference.assert_called_once_with("qwen3_8b", str(wf_linked.id))
    # 登记引用只挡 idle/LRU 卸载,绝不触发加载(2026-09-03 删 _load_wf_deps 预热)。
    mm.load_model.assert_not_called()
```

改为

```python
    # spec 2026-09-05 §7:已发布工作流不再登记进程级引用 —— 那等于把 resident:false
    # 的模型变成事实常驻且从 /api/v1/engines 看不出来(2026-09-05「新工作流」钉死 qwen3_6)。
    mm.add_reference.assert_not_called()
    mm.load_model.assert_not_called()
```

`backend/tests/test_startup_model_load_policy.py` 第 107-109 行,把

```python
    assert (("qwen3_6_35b_a3b_fp8", str(wf.id))) in refs, \
        "published 工作流的模型引用仍应被登记(防 idle/LRU 卸载)"
```

改为

```python
    assert refs == [], "spec 2026-09-05 §7:已发布工作流不再登记模型引用,resident 是唯一钉住手段"
```

- [ ] **Step 2: 写 TTL 回收测试**

新建 `backend/tests/test_idle_ttl_ignores_workflows.py`:

```python
"""spec 2026-09-05 §7:非常驻模型即使被已发布工作流"依赖",只要没有活跃引用就按 TTL 回收;
请求期的 proxy_ref 仍能挡住回收。直接驱动 ModelManager.check_idle_models,不起任何子进程。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services.model_manager import ModelManager


def _mgr_with(entry_last_used: float, refs: set[str]) -> ModelManager:
    mgr = ModelManager.__new__(ModelManager)          # 绕过 __init__(它会拉 GPU 探测)
    mgr._models = {
        "qwen3_6_35b_a3b_fp8": SimpleNamespace(
            spec=SimpleNamespace(resident=False, ttl_seconds=300),
            last_used=entry_last_used,
            adapter=SimpleNamespace(is_loaded=True),
        )
    }
    mgr._references = {"qwen3_6_35b_a3b_fp8": set(refs)}
    mgr.unload_model = AsyncMock(return_value=True)
    return mgr


@pytest.mark.asyncio
async def test_idle_non_resident_model_is_reclaimed_without_refs(monkeypatch):
    import time
    monkeypatch.setattr(time, "monotonic", lambda: 10_000.0)
    mgr = _mgr_with(entry_last_used=10_000.0 - 301, refs=set())
    await mgr.check_idle_models()
    mgr.unload_model.assert_awaited_once_with("qwen3_6_35b_a3b_fp8")


@pytest.mark.asyncio
async def test_proxy_ref_still_blocks_reclaim(monkeypatch):
    import time
    monkeypatch.setattr(time, "monotonic", lambda: 10_000.0)
    mgr = _mgr_with(entry_last_used=10_000.0 - 301, refs={"proxy-deadbeef"})
    await mgr.check_idle_models()
    mgr.unload_model.assert_not_awaited()
```

- [ ] **Step 3: 跑,确认失败面正确**

Run: `cd backend && uv run pytest tests/test_startup_reconcile.py tests/test_startup_model_load_policy.py tests/test_idle_ttl_ignores_workflows.py -n 8 -q`
Expected: `test_orphan_published_reverts_to_draft_and_linked_registers_deps` 与 `test_startup_model_load_policy` 里翻转的那条 FAIL(`add_reference` 仍被调);`test_idle_ttl_ignores_workflows.py` 两条 PASS(它们验证的是 `check_idle_models` 既有行为,作为回归锚)。

- [ ] **Step 4: 改 `startup_reconcile.py`**

把 `reconcile_orphan_published_workflows` 的循环体

```python
    for wf in published:
        if wf.id not in linked_wf_ids:
            wf.status = "draft"
            orphan += 1
            continue
        for dep in model_mgr.get_model_dependencies({"nodes": wf.nodes, "edges": wf.edges}):
            model_mgr.add_reference(dep["key"], str(wf.id))
```

改为

```python
    for wf in published:
        if wf.id not in linked_wf_ids:
            wf.status = "draft"
            orphan += 1
```

并把 docstring 里「有关联的重登记模型引用(防常驻模型被 idle/LRU 卸)」一句及其后「**登记引用 ≠ 加载模型**…」整段替换为:

```
    2026-09-05(spec engine-app-boundary §7):不再给已发布工作流的模型依赖登记进程级
    引用。那条引用只在下架/删除时移除,效果是 resident:false 的模型一旦被加载就永不
    TTL 回收(真机:「新工作流」把 qwen3_6 钉死在 3090 对上),且 /api/v1/engines 看不出
    原因。想让工作流依赖的模型常热,写 resident: true;请求期保护仍由 proxy_ref 与
    _in_use 负责。参数 model_mgr 保留以免改所有调用点签名,本函数已不使用它。
```

- [ ] **Step 5: 改 `workflows.py` 下架**

`unpublish_workflow` 里删除这一段(`# Remove model references and attempt unload` 起的 6 行):

```python
    # Remove model references and attempt unload
    deps = model_mgr.get_model_dependencies(
        {"nodes": wf.nodes, "edges": wf.edges}
    )
    for dep in deps:
        model_mgr.remove_reference(dep["key"], str(wf.id))
        await model_mgr.unload_model(dep["key"])
```

函数开头 `model_mgr = request.app.state.model_manager` 一并删除(若删后 `request` 参数无其他用途,保留参数不动,避免改路由签名)。

- [ ] **Step 6: 改 `services.py` 删服务**

把 `if still.first() is None:` 分支里 `model_mgr = getattr(...)` 起到 `except Exception as e:` 日志结束的整段删除,只保留:

```python
        if still.first() is None:
            wf = await session.get(Workflow, wf_id)
            if wf is not None and wf.status == "published":
                wf.status = "draft"
```

并把该段上方注释「顺带卸掉该工作流的模型引用。」删掉。

- [ ] **Step 7: 跑,确认通过 + 相关面无回归**

Run: `cd backend && uv run pytest tests/test_startup_reconcile.py tests/test_startup_model_load_policy.py tests/test_idle_ttl_ignores_workflows.py tests/test_service_autostart.py tests/test_model_manager_v2.py -n 8 -q`
Expected: 全部 PASS。
Run: `cd backend && grep -rn "remove_reference\|add_reference" src/ --include='*.py' | grep -v "def "`
Expected: 只剩 `src/api/routes/openai_compat.py` 的 `proxy_ref` 三处。

- [ ] **Step 8: 提交、PR**

```bash
git checkout -b fix/no-workflow-model-pin master
git add backend/src/services/startup_reconcile.py backend/src/api/routes/workflows.py backend/src/api/routes/services.py backend/tests/test_startup_reconcile.py backend/tests/test_startup_model_load_policy.py backend/tests/test_idle_ttl_ignores_workflows.py
git -c commit.gpgsign=false commit -m "fix(model-manager): 已发布工作流不再钉住模型 —— resident 是唯一常驻手段

spec 2026-09-05 §7。startup_reconcile 给已发布工作流依赖登记的进程级引用,让
resident:false 的模型一旦加载就永不 TTL 回收(真机:「新工作流」钉死 qwen3_6 占满
3090 对),且 /api/v1/engines 看不出原因。请求期保护仍由 proxy_ref 与 _in_use 负责。"
git push -u origin fix/no-workflow-model-pin && gh pr create --base master --fill
```

---

### Task 3: 关门 + 快失败错误契约 + `/v1/models` 只列就绪(PR-3)

**Files:**
- Modify: `backend/src/errors.py`(加 `ModelNotReadyError`)
- Modify: `backend/src/services/inference/vllm_endpoint.py:64-88`(删 `ensure_vllm_base_url`)、`:31-62`(错误文案)
- Create: `backend/src/api/routes/_readiness.py`
- Modify: `backend/src/api/routes/openai_compat.py`(`:36-40` import、`:246-252` chat、`:585-591` embeddings、`:722-736` `_resolve_moss_base_url`、`:1533-1540` `ModelObject`、`:1565-1610` `/v1/models`)
- Modify: `backend/src/api/routes/anthropic_compat.py:54-57,171-177`、`ollama_compat.py:33-36,93-99`、`responses.py:45-48,376-382`、`context_cache.py:36-39,87-93`
- Test: 新增 `backend/tests/test_data_plane_readonly.py`;改 `tests/test_compat_routes_vllm_regression.py:20-45`、`tests/test_embeddings_endpoint.py:41`、`tests/test_vllm_endpoint.py:59-118`、`tests/test_asr_transcription.py:560-590,795-806`、`tests/test_v1_models_scoped.py`

**Interfaces:**
- Produces: `src.errors.ModelNotReadyError(model: str, *, ready_models: list[str] | None = None)` → 503,`type=code="model_not_ready"`,`param="model"`,可选 `error.ready_models`。
- Produces: `src.api.routes._readiness.engine_name_of(svc) -> str`、`service_is_ready(model_mgr, svc) -> bool`、`ready_model_names(model_mgr, services) -> list[str]`。
- Consumes: `get_vllm_base_url(model_mgr, engine_name) -> str`(只读)、`ModelManager.is_loaded(name)`(**不** `touch()`,不刷新 TTL —— 别用 `get_adapter` 判就绪)。

- [ ] **Step 1: 静态守卫测试 + 快失败测试(先失败)**

新建 `backend/tests/test_data_plane_readonly.py`:

```python
"""spec 2026-09-05 §4/§5/§6:数据面对模型放置只读。

1) 静态守卫:五个兼容路由模块不得引用任何加载能力(与 _placement.py 三条不变式同路数)。
2) 行为:已授权但未加载 → 503 model_not_ready,且 load_model 未被调用。
3) /v1/models 只列就绪的 model 服务;非 model 服务照旧;owned_by == "nous-engine"。
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import bcrypt
import pytest

from src.models.api_gateway import ApiKeyGrant
from src.models.instance_api_key import InstanceApiKey
from src.models.service_instance import ServiceInstance

DATA_PLANE_MODULES = (
    "src.api.routes.openai_compat",
    "src.api.routes.anthropic_compat",
    "src.api.routes.ollama_compat",
    "src.api.routes.responses",
    "src.api.routes.context_cache",
)


def test_data_plane_modules_have_no_load_capability():
    import importlib
    for name in DATA_PLANE_MODULES:
        src = inspect.getsource(importlib.import_module(name))
        assert "ensure_vllm_base_url" not in src, f"{name} 仍引用懒加载 helper"
        assert ".load_model(" not in src, f"{name} 直接调 load_model —— 数据面不得改变放置"
        assert "get_vllm_base_url" in src, f"{name} 应只经只读的 get_vllm_base_url 解析端点"


def test_ensure_vllm_base_url_is_gone():
    import src.services.inference.vllm_endpoint as ve
    assert not hasattr(ve, "ensure_vllm_base_url")


def _hash(t: str) -> str:
    return bcrypt.hashpw(t.encode(), bcrypt.gensalt()).decode()


async def _model_svc(db_session, name, engine, category="llm"):
    s = ServiceInstance(source_type="model", source_name=engine, name=name,
                        type="llm", status="active", category=category)
    db_session.add(s)
    await db_session.commit()
    await db_session.refresh(s)
    return s


async def _grant_key(db_session, *services):
    raw = "sk-ready1234abcdef"
    k = InstanceApiKey(instance_id=None, label="t", key_hash=_hash(raw),
                       key_prefix=raw[:10], is_active=True)
    db_session.add(k)
    await db_session.commit()
    await db_session.refresh(k)
    db_session.add_all([ApiKeyGrant(api_key_id=k.id, service_id=s.id, status="active") for s in services])
    await db_session.commit()
    return raw


@pytest.mark.asyncio
async def test_chat_on_cold_model_is_503_model_not_ready_and_never_loads(db_client, db_session):
    cold = await _model_svc(db_session, "qwen3-8-27b", "qwen3_8_27b_abliterated_awq")
    hot = await _model_svc(db_session, "moss-asr", "moss_transcribe_diarize", category="asr")
    raw = await _grant_key(db_session, cold, hot)
    mgr = db_client._transport.app.state.model_manager
    mgr.get_adapter = MagicMock(return_value=None)                       # 冷:没有 adapter
    mgr.is_loaded = MagicMock(side_effect=lambda n: n == "moss_transcribe_diarize")

    r = await db_client.post(
        "/v1/chat/completions",
        json={"model": "qwen3-8-27b", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 503, r.text
    err = r.json()["error"]
    assert err["type"] == "model_not_ready" and err["code"] == "model_not_ready"
    assert err["param"] == "model"
    assert err["ready_models"] == ["moss-asr"]
    mgr.load_model.assert_not_called()   # 数据面绝不加载


@pytest.mark.asyncio
async def test_streaming_chat_on_cold_model_is_plain_503(db_client, db_session):
    cold = await _model_svc(db_session, "qwen3-8-27b", "qwen3_8_27b_abliterated_awq")
    raw = await _grant_key(db_session, cold)
    mgr = db_client._transport.app.state.model_manager
    mgr.get_adapter = MagicMock(return_value=None)
    r = await db_client.post(
        "/v1/chat/completions",
        json={"model": "qwen3-8-27b", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")   # 开流前拒绝,不是断掉的 SSE
    mgr.load_model.assert_not_called()


@pytest.mark.asyncio
async def test_v1_models_lists_only_ready_model_services(db_client, db_session):
    cold = await _model_svc(db_session, "qwen3-8-27b", "qwen3_8_27b_abliterated_awq")
    hot = await _model_svc(db_session, "moss-asr", "moss_transcribe_diarize", category="asr")
    wf = ServiceInstance(source_type="workflow", source_name="x", name="krea2",
                         type="inference", status="active", category="image")
    db_session.add(wf)
    await db_session.commit()
    await db_session.refresh(wf)
    raw = await _grant_key(db_session, cold, hot, wf)
    mgr = db_client._transport.app.state.model_manager
    mgr.is_loaded = MagicMock(side_effect=lambda n: n == "moss_transcribe_diarize")

    r = await db_client.get("/v1/models", headers={"Authorization": f"Bearer {raw}"})
    assert r.status_code == 200, r.text
    data = {m["id"]: m for m in r.json()["data"]}
    assert set(data) == {"moss-asr", "krea2"}           # 冷的 model 服务不出现;workflow 服务照旧
    assert all(m["owned_by"] == "nous-engine" for m in data.values())

    one = await db_client.get("/v1/models/qwen3-8-27b", headers={"Authorization": f"Bearer {raw}"})
    assert one.status_code == 503 and one.json()["error"]["code"] == "model_not_ready"
    missing = await db_client.get("/v1/models/nope", headers={"Authorization": f"Bearer {raw}"})
    assert missing.status_code == 404
```

Run: `cd backend && uv run pytest tests/test_data_plane_readonly.py -q`
Expected: 全部 FAIL(守卫:`ensure_vllm_base_url` 仍在;行为:`ModelNotReadyError` 不存在 / 503 体没有 `ready_models` / `/v1/models` 列出了冷模型)。

- [ ] **Step 2: `ModelNotReadyError`**

`backend/src/errors.py` 末尾追加:

```python
class ModelNotReadyError(ServiceUnavailableError):
    """503 — 模型已授权给该 key 但当前未加载(含 loading 中)。

    spec 2026-09-05 §5:数据面对放置只读,未就绪即刻拒绝,绝不在请求路径上加载。
    `ready_models` = 该 key 已授权 ∩ 当前已加载 的服务名(调用方能拿到时才带)。
    """

    type = "model_not_ready"

    def __init__(self, model: str, *, ready_models: list[str] | None = None, **kw):
        super().__init__(
            f"model '{model}' is not loaded; see GET /v1/models for ready models",
            code="model_not_ready",
            param="model",
            fix="GET /v1/models lists the models that are ready right now; loading is a control-plane action",
            **kw,
        )
        self.ready_models = ready_models

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.ready_models is not None:
            d["error"]["ready_models"] = list(self.ready_models)
        return d
```

- [ ] **Step 3: readiness helper**

新建 `backend/src/api/routes/_readiness.py`:

```python
"""服务就绪判定 —— /v1/models 与 model_not_ready 错误体共用,保证两处口径一致。

spec 2026-09-05 §5/§6。只看 ModelManager.is_loaded(不用 get_adapter:它会 touch()
刷新 last_used,列一次模型就把 TTL 续了)。非 model 类服务(comfy_template / workflow /
app)不占 nous-engine 显存,一律视为就绪 —— 它们的就绪由各自执行路径负责。
"""
from __future__ import annotations

from typing import Any, Iterable


def engine_name_of(svc: Any) -> str:
    """服务名 ↔ 引擎名的唯一写法(与 chat/embeddings 路由一致)。"""
    return svc.source_name or str(svc.source_id)


def service_is_ready(model_mgr: Any, svc: Any) -> bool:
    if getattr(svc, "source_type", None) != "model":
        return True
    if model_mgr is None:
        return False
    return bool(model_mgr.is_loaded(engine_name_of(svc)))


def ready_model_names(model_mgr: Any, services: Iterable[Any]) -> list[str]:
    """该批服务里「model 类且已加载」的服务名,按传入顺序。"""
    return [s.name for s in services
            if getattr(s, "source_type", None) == "model" and service_is_ready(model_mgr, s)]
```

- [ ] **Step 4: 删 `ensure_vllm_base_url`,改只读文案**

`backend/src/services/inference/vllm_endpoint.py`:删除第 64-88 行整个 `ensure_vllm_base_url` 函数;`get_vllm_base_url` 里 `VLLMNotLoaded` 的文案改为 `f"模型 '{engine_name}' 未加载"`(去掉「请在模型管理页加载后重试」——这句是给 admin 看的,现在这条路是给下游看的);模块 docstring 加一行:「2026-09-05 spec §4:删除按需懒加载变体 `ensure_vllm_base_url`,数据面对放置只读,加载只经 `/api/v1/engines/{name}/load`」。

- [ ] **Step 5: 六处调用改只读 + 抛 `ModelNotReadyError`**

每个文件先改 import:把 `ensure_vllm_base_url,` 从 `from src.services.inference.vllm_endpoint import (...)` 里删掉(`openai_compat.py` 同时保留已有的 `get_vllm_base_url`;其余四个文件把 `ensure_vllm_base_url` 换成 `get_vllm_base_url`)。再加 `from src.errors import ModelNotReadyError`(`responses.py`/`context_cache.py` 已从 `src.api.deps_auth` 引 `APIError`,在同一行追加 `ModelNotReadyError` 若该模块 re-export;否则直接 `from src.errors import ModelNotReadyError`)。

`openai_compat.py` chat(原 246-252):

```python
    model_mgr = getattr(request.app.state, "model_manager", None)
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        from src.api.routes._readiness import ready_model_names  # noqa: PLC0415
        raise ModelNotReadyError(
            body.get("model") or engine_name,
            ready_models=ready_model_names(model_mgr, await _granted_services(session, api_key)),
        ) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e
```

`session` / `api_key` 是该 handler 已有的依赖注入变量(`auth` 元组解出的 `api_key`,与 `list_models` 同款);用 `sed -n '180,236p' src/api/routes/openai_compat.py` 确认名字后再写。若 handler 没有 `session` 参数,加 `session: AsyncSession = Depends(get_async_session)`(文件已 import 两者)。

`openai_compat.py` embeddings(原 585-591)同上,`ready_models` 同样算。

`anthropic_compat.py`(原 171-177)、`ollama_compat.py`(原 93-99):

```python
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise ModelNotReadyError(engine_name) from e
    except VLLMNoEndpoint as e:
        raise HTTPException(500, detail=str(e)) from e
```

`responses.py`(原 376-382)、`context_cache.py`(原 87-93):

```python
    try:
        base_url = get_vllm_base_url(model_mgr, engine_name)
    except VLLMNotLoaded as e:
        raise ModelNotReadyError(engine_name) from e
    except VLLMNoEndpoint as e:
        raise APIError(str(e), code="no_inference_endpoint") from e
```

(这四处的 key 授权范围不在手边,`ready_models` 省略;错误体的 type/code/fix 与 chat 完全一致。)

`openai_compat.py` `_resolve_moss_base_url`(原 722-736):`return (await ensure_vllm_base_url(model_mgr, engine_name)).rstrip("/")` 改为 `return get_vllm_base_url(model_mgr, engine_name).rstrip("/")`;函数改为普通 `def`(不再 await);docstring 里「未加载则按需拉起再解析」改为「只读:MOSS 是 resident,未加载即 VLLMNotLoaded(调用方映射 503),不在请求路径上拉起」。用 `grep -n "_resolve_moss_base_url" src/api/routes/openai_compat.py` 找到调用点,去掉那处的 `await`。

- [ ] **Step 6: `/v1/models` 只列就绪 + `owned_by`**

`ModelObject.owned_by: str = "nous-center"` → `"nous-engine"`。

`list_models` 加参数 `request: Request`,主体改为:

```python
    _instance, api_key = auth
    if api_key is None:
        raise NotFoundError("request requires an API key", code="model_not_found")
    from src.api.routes._readiness import service_is_ready  # noqa: PLC0415
    model_mgr = getattr(request.app.state, "model_manager", None)
    services = await _granted_services(session, api_key)
    data = [
        ModelObject(id=s.name, type=(s.category or "model"))
        for s in services
        if (not type or (s.category or "model") == type)
        and service_is_ready(model_mgr, s)      # spec 2026-09-05 §6:只列现在就能调的
    ]
    return ModelListResponse(data=data)
```

docstring 追加:「2026-09-05 起 model 类服务只在其模型已加载时出现(`/v1/models` = 现在就能调的);comfy_template / workflow / app 照旧按授权列。」

`get_model` 加 `request: Request`,在 `if svc is None: raise NotFoundError(...)` 之后加:

```python
    from src.api.routes._readiness import service_is_ready  # noqa: PLC0415
    if not service_is_ready(getattr(request.app.state, "model_manager", None), svc):
        raise ModelNotReadyError(model_id)      # 「没就绪」(503)与「没授权」(404)分开
```

- [ ] **Step 7: 更新被删函数牵连的既有测试**

- `tests/test_compat_routes_vllm_regression.py:20-45`:断言改为 `assert "get_vllm_base_url" in content` 且 `assert "ensure_vllm_base_url" not in content`;注释改为「2026-09-05 spec §4:数据面只读,懒加载变体已删除」。
- `tests/test_embeddings_endpoint.py:41`:`assert "ensure_vllm_base_url" in blk` → `assert "get_vllm_base_url" in blk`,提示文案改「只读 base_url 查找」。
- `tests/test_vllm_endpoint.py`:删掉第 7 行 import 里的 `ensure_vllm_base_url,` 与第 59-118 行整段「按需懒加载」用例;在文件末尾加:

```python
def test_no_lazy_load_variant_exists():
    """spec 2026-09-05 §4:数据面唯一的加载门已拆。"""
    import src.services.inference.vllm_endpoint as ve
    assert not hasattr(ve, "ensure_vllm_base_url")
```

- `tests/test_asr_transcription.py:560-590`(两条 `_resolve_moss_base_url` 用例):`monkeypatch.setattr(oc, "ensure_vllm_base_url", ...)` 改为 `monkeypatch.setattr(oc, "get_vllm_base_url", ...)`,对应 fake 从 `async def` 改成 `def`;调用处 `await _resolve_moss_base_url(...)` 改成不 await。
- `tests/test_asr_transcription.py:795-806`:删掉 `_boom_ensure` 与那句 `monkeypatch.setattr(oc, "ensure_vllm_base_url", _boom_ensure)`(函数已不存在,`setattr` 会 AttributeError);`_fake_get` 里默认引擎断言此 Task 先不动(Task 4 改)。
- `tests/test_v1_models_scoped.py`:两条既有用例用的是 `source_type="workflow"`,不受影响;不改。
- 运行 `cd backend && uv run pytest tests/test_api_keys_global.py -q`;若某条因 model 类服务被就绪过滤而失败,在该用例开头加 `client._transport.app.state.model_manager.is_loaded = MagicMock(return_value=True)`(它测的是授权,不是就绪)。

- [ ] **Step 8: 跑,确认通过 + 回归面**

Run: `cd backend && uv run pytest tests/test_data_plane_readonly.py tests/test_compat_routes_vllm_regression.py tests/test_embeddings_endpoint.py tests/test_vllm_endpoint.py tests/test_asr_transcription.py tests/test_v1_models_scoped.py tests/test_anthropic_compat.py tests/test_ollama_compat.py tests/test_responses_service.py tests/test_context_cache_service.py tests/test_chat_completions_dispatch.py tests/test_api_keys_global.py -n 8 -q`
Expected: 全部 PASS。
Run: `cd backend && grep -rn "ensure_vllm_base_url" src tests`
Expected: 无输出。

- [ ] **Step 9: 全量回归**

Run: `cd backend && uv run pytest tests -n 8 -q`(约 65s)
Expected: 全绿。

- [ ] **Step 10: 提交、PR**

```bash
git checkout -b feat/data-plane-readonly master
git add backend/src/errors.py backend/src/services/inference/vllm_endpoint.py backend/src/api/routes/_readiness.py backend/src/api/routes/openai_compat.py backend/src/api/routes/anthropic_compat.py backend/src/api/routes/ollama_compat.py backend/src/api/routes/responses.py backend/src/api/routes/context_cache.py backend/tests/test_data_plane_readonly.py backend/tests/test_compat_routes_vllm_regression.py backend/tests/test_embeddings_endpoint.py backend/tests/test_vllm_endpoint.py backend/tests/test_asr_transcription.py
git -c commit.gpgsign=false commit -m "feat(api): 数据面对模型放置只读 —— 未就绪即刻 503 model_not_ready,/v1/models 只列就绪

spec 2026-09-05 §4/§5/§6。删除数据面唯一的懒加载入口 ensure_vllm_base_url,五个兼容
路由只经只读 get_vllm_base_url;静态守卫测试锁住。此前下游一次请求会阻塞 85-111s
冷加载并把模型落到别人钉死的卡上。owned_by 改 nous-engine。"
git push -u origin feat/data-plane-readonly && gh pr create --base master --fill
```

---

### Task 4: 目录落地 + 常驻容量测试 + CLAUDE.md(PR-4)

**Files:**
- Modify: `backend/configs/models.d/qwen3_8_27b_abliterated_awq.yaml`(`resident: false` → `true`)
- Modify: `backend/configs/models.d/qwen3_embedding_8b.yaml`(加 `gpu: 1`、`resident: true`)
- Delete: `backend/configs/models.d/qwen3_6_35b_a3b_fp8.yaml`
- Modify: `backend/src/api/routes/openai_compat.py:821,839`(`_PUNCT_LLM_ENGINE_DEFAULT`)
- Modify: `backend/tests/test_asr_transcription.py:802`
- Create: `backend/tests/test_resident_capacity.py`
- Modify: `CLAUDE.md`(「GPU 放置」节加不变式;修正 `/v1/*` 鉴权回退那句)

**Interfaces:**
- Consumes: `src.services.model_scanner.scan_models() -> dict[str, dict]`、`src.gpu.topology.resolve_gpus(cfg) -> list[int]`、`src.config.load_hardware_config() -> {"groups": [...]}`、`src.services.gpu_monitor.DEFAULT_RESERVED_GB`。

- [ ] **Step 1: 写容量测试(先失败:embedding 没钉卡)**

新建 `backend/tests/test_resident_capacity.py`:

```python
"""spec 2026-09-05 §10:所有 resident: true 的模型按落卡汇总 vram_mb,必须放得进
hardware.yaml 的容量减 DEFAULT_RESERVED_GB。常驻集合不自洽在合 PR 前就红,不等上线。

不碰 nvidia-smi:容量来自 hardware.yaml 里单卡 group 的 vram_gb(GPU 0/2 各 24、GPU 1 96)。
"""
from __future__ import annotations

from collections import defaultdict

from src.config import load_hardware_config
from src.gpu.topology import resolve_gpus
from src.services.gpu_monitor import DEFAULT_RESERVED_GB
from src.services.model_scanner import scan_models


def _capacity_gb_by_gpu() -> dict[int, float]:
    cap: dict[int, float] = {}
    for g in load_hardware_config().get("groups", []):
        gpus = list(g.get("gpus") or [])
        if len(gpus) == 1:
            cap[gpus[0]] = float(g["vram_gb"])
    return cap


def test_resident_models_are_pinned_and_fit():
    cap = _capacity_gb_by_gpu()
    assert cap, "hardware.yaml 没有单卡 group,无法推容量"
    used = defaultdict(float)
    for key, cfg in scan_models().items():
        if not cfg.get("resident"):
            continue
        gpus = resolve_gpus(cfg)
        assert gpus, f"常驻模型 {key} 必须显式钉卡(gpu/gpus),常驻不能靠自动选卡"
        share = float(cfg.get("vram_mb", 0)) / 1024 / len(gpus)
        for g in gpus:
            used[g] += share
    for g, total in used.items():
        assert g in cap, f"GPU {g} 不在 hardware.yaml 里"
        assert total <= cap[g] - DEFAULT_RESERVED_GB, (
            f"GPU {g} 常驻合计 {total:.1f}G 超过 {cap[g]}G - 预留 {DEFAULT_RESERVED_GB}G"
        )


def test_qwen3_6_is_retired():
    assert "qwen3_6_35b_a3b_fp8" not in scan_models(), "spec §10:3.6 退役,目录唯一 LLM 是 3.8"
```

Run: `cd backend && uv run pytest tests/test_resident_capacity.py -q`
Expected: `test_qwen3_6_is_retired` FAIL(yaml 还在);`test_resident_models_are_pinned_and_fit` 此时 PASS 或因 scan 到的 resident 集合不同而失败均可——改完配置后必须 PASS。

- [ ] **Step 2: 改三个 yaml**

`qwen3_8_27b_abliterated_awq.yaml`:`resident: false` → `resident: true`,上方加注释:

```yaml
# 2026-09-05 spec engine-app-boundary §10:目录唯一 LLM,常驻。数据面不再懒加载,
# 不常驻 = 下游(ComfyUI 反推 / nous-app)一律 503 model_not_ready。
```

`qwen3_embedding_8b.yaml`:在 `vram_mb: 17500` 之后加:

```yaml
# 2026-09-05 spec §10:钉 Pro 6000 并常驻。此前无钉卡走自动选卡,反复落到 3090 又被
# LRU 驱逐(gpu_monitor 日志 "Auto-evicted model qwen3_embedding_8b from GPU 2")。
gpu: 1
resident: true
```

`git rm backend/configs/models.d/qwen3_6_35b_a3b_fp8.yaml`(权重目录 `llm/Qwen3.6-35B-A3B-FP8` 留盘不动)。

- [ ] **Step 3: 标点 LLM 默认引擎改 3.8**

`openai_compat.py:821`:`_PUNCT_LLM_ENGINE_DEFAULT = "qwen3_8_27b_abliterated_awq"`;第 839 行 docstring 里的默认名同步。`tests/test_asr_transcription.py:802`:`assert name == "qwen3_8_27b_abliterated_awq"  # 默认引擎(2026-09-05 3.6 退役)`。

- [ ] **Step 4: 跑,确认通过**

Run: `cd backend && uv run pytest tests/test_resident_capacity.py tests/test_asr_transcription.py tests/test_api_engines.py tests/test_model_delete.py tests/test_service_autostart.py -n 8 -q`
Expected: 全部 PASS(`test_model_delete` / `test_service_autostart` 只把 `qwen3_6_35b_a3b_fp8` 当字符串数据用,不依赖 yaml)。

- [ ] **Step 5: CLAUDE.md**

「GPU 放置 / 张量并行」节末尾加一条:

```
- **模型放置只能由控制面改变;数据面(`/v1/*` 全部兼容路由)对放置只读**(spec 2026-09-05
  engine-app-boundary)。未加载的模型一律即刻 503 `model_not_ready`,不在请求路径上
  加载;`/v1/models` 只列已加载的 model 类服务;`resident: true` 是**唯一**的常驻手段,
  已发布工作流不再钉住模型。`tests/test_data_plane_readonly.py` 静态锁住五个路由模块。
```

「ComfyUI 桥」节里「`/v1/*` 带了 `Authorization` header 优先走 M:N `InstanceApiKey` 校验,校验失败(含 `ADMIN_TOKEN`…)会退回 admin-session 校验」后加括号:「(**`/v1/audio/transcriptions` 例外**:`_auth_transcriptions` 直连 `verify_bearer_token_any`,`ADMIN_TOKEN` 会被拒为 Invalid API key,脚本化调用要铸 InstanceApiKey —— 2026-09-05 实测)」。

- [ ] **Step 6: 提交、PR**

```bash
git checkout -b config/catalog-resident-3.8 master
git add backend/configs/models.d/ backend/src/api/routes/openai_compat.py backend/tests/test_asr_transcription.py backend/tests/test_resident_capacity.py CLAUDE.md
git -c commit.gpgsign=false commit -m "config(models): 目录落地 —— 3.8 常驻 3090 对,embedding 钉 Pro 6000 常驻,3.6 退役

spec 2026-09-05 §10。新增 test_resident_capacity:常驻集合按落卡汇总必须放得进
hardware.yaml 容量减预留。标点 LLM 默认引擎随之改 3.8。"
git push -u origin config/catalog-resident-3.8 && gh pr create --base master --fill
```

合并后**必须重启后端**让常驻生效:`sudo systemctl restart nous-engine-backend`(需要用户在真终端执行);重启后 `curl -s -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:8000/api/v1/engines | python3 -c "import json,sys;[print(e['name'],e['status'],e['loaded_gpus'],e['held_by']) for e in json.load(sys.stdin) if e['status']!='unloaded']"` 应列出 `qwen3_8_27b_abliterated_awq [0, 2]`、`qwen3_embedding_8b [1]`、`moss_transcribe_diarize [1]`,`held_by` 全为 `[]`。

---

### Task 5: 真机验收(spec §12,非 CI)

**Files:** 无代码改动;产出一份验收记录追加到 spec §12 末尾(`docs/superpowers/specs/2026-09-05-engine-app-boundary-data-plane-readonly-design.md`),并提 docs PR。

前置:Task 1-4 全部合并、后端已重启。所有 curl 前先 `unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy`,`ADMIN_TOKEN` 取自 `backend/.env`,ComfyUI 用的 key 在 `ComfyUI/user/default/prompt-assistant/config/config.json` 的 `nous` 服务里。

- [ ] **Step 1: 就绪即服务**

```bash
KEY=<nous 服务的 api_key>
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models
```
Expected: `data` 含 `{"id":"qwen3-8-27b","owned_by":"nous-engine",...}`。再用 ComfyUI 跑一次 `Krea2-全能总控-11模式.json` 反推开关=true(scratchpad 已有 `smoke_caption.json`),出图成功。

- [ ] **Step 2: 未就绪即拒绝、且无副作用**

```bash
curl -s -X POST -H "Authorization: Bearer $ADMIN_TOKEN" "http://127.0.0.1:8000/api/v1/engines/qwen3_8_27b_abliterated_awq/unload?force=true"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader      # 记下 GPU 0/2 数字
time curl -s -o /tmp/claude-1000/nr.json -w '%{http_code}\n' -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8-27b","messages":[{"role":"user","content":"hi"}]}' http://127.0.0.1:8000/v1/chat/completions
cat /tmp/claude-1000/nr.json
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader      # 必须与上面相同
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models   # 不含 qwen3-8-27b
curl -s -X POST -H "Authorization: Bearer $ADMIN_TOKEN" http://127.0.0.1:8000/api/v1/engines/qwen3_8_27b_abliterated_awq/load
```
Expected:`503`,`real < 0.1s`,body `error.code == "model_not_ready"` 且 `ready_models` 含 `moss-asr`;GPU 0/2 显存前后一致;`/v1/models` 不含 3.8;显式 load 后恢复。

- [ ] **Step 3: ComfyUI 人像整条跑通(用户硬性要求)**

已于 2026-09-05 完成一次(`ComfyUI_00013_.png` 5632×3072,Aiden 三段流水线 259s,反推经 nous-engine 200)。Task 4 重启后再跑一次确认常驻状态下同样成立:`mcp__comfy-local__run_workflow` 跑 scratchpad 的 `wf3_full_switch_on.json`(源图 `smoke_caption_src.png`),`job(action="wait", timeout_seconds=1800)`,产物尺寸 5632×3072,Read 缩略图确认写实人像。

- [ ] **Step 4: 下游只反映状态(观察项)**

```bash
docker exec nous-db psql -U postgres -p 55434 -d postgres -c "SELECT name,actual_model,last_test_status,last_tested_at FROM public.mediahub_models WHERE base_url LIKE '%8000%';"
```
Expected:3.8 加载时 `nous-qwen3-llm` 那行仍 fail(它指向已退役的 `qwen3-6-35b`,这是 nous-app 自己的目录问题,§13 范围外);`mediahub-moss-asr` ok;火山/deepseek 各行不受影响。

- [ ] **Step 5: 把 Step 1-4 的实测数字追加到 spec §12,提交 docs PR**

```bash
git checkout -b docs/engine-app-boundary-acceptance master
git add docs/superpowers/specs/2026-09-05-engine-app-boundary-data-plane-readonly-design.md
git -c commit.gpgsign=false commit -m "docs(spec): engine-app-boundary 真机验收记录"
git push -u origin docs/engine-app-boundary-acceptance && gh pr create --base master --fill
```
