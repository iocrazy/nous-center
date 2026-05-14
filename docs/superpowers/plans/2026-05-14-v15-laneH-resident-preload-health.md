# V1.5 Lane H: resident preload_order + _load_failures + /health 扩展 + GPU-free gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 V1.5 的「鲁棒 resident preload」三件事落地：(1) `models.yaml` 加 `preload_order: int`，启动时按它升序 preload resident 模型，单个失败 fail-soft（写 `_load_failures`，不阻断 API server 启动）；(2) `/health` 端点扩展，暴露 per-runner 状态 + `_load_failures`，让 Dashboard 能显示 degraded banner；(3) F2 GPU-free gate —— 把 Lane C `RunnerSupervisor._default_gpu_free_probe` 那个「无 GPU 保守返回 True」的骨架，换成真的 `nvidia-smi`-backed 探针：runner 重启后必须等该 group 的 GPU 显存回落到基线才 re-preload（死进程 CUDA context 回收是异步的，纯 backoff 可能太短）。

**Architecture:** 四块改动，互相独立、可分别 commit：

1. **`ModelSpec` + registry 加 `preload_order`。** `ModelSpec`（`registry.py`）现在有 `resident: bool` 但没有 `preload_order`。Lane H 加 `preload_order: int | None = None` 字段，`_load`/`add_from_scan` 都从 yaml 读。spec §3.2 的 models.yaml 草图明确列了 `preload_order`，但当前 `configs/models.yaml` 还没有这个字段（已 grep 确认全是 `resident: false`，无 `preload_order`）—— Lane H 加字段定义 + 给现有 resident 模型补 yaml 值（当前实际无 resident 模型，所以是纯字段定义 + 默认值语义）。

2. **`ModelManager.preload_residents()` —— 按 `preload_order` 升序 fail-soft preload。** 当前 `main.py` lifespan 里有一段 `_preload_image_model` 背景 task：它**只 preload image 类 resident 模型**、**无序**（按 `registry.specs` 列表顺序）、失败写 `_load_failures`。Lane H 把这块「按 preload_order 排序 + 跨 type fail-soft 遍历」的逻辑收进 `ModelManager.preload_residents()` 一个方法，`main.py` 改调它。排序规则（spec §5.2「`preload_order` 排序」测试要点）：`resident:true` 的模型按 `preload_order` 升序；`preload_order` 为 `None` 的排最后（FIFO 兜底）。单个 `load_model` 抛异常 → 写 `_load_failures[model_id]` + 继续下一个，**绝不向上抛**。

3. **`/health` 扩展 —— per-runner 状态 + `_load_failures`。** `main.py` 的 `create_app()` 里有个 inline `/health`：现在返回 `{status, database, gpus, models_loaded}`。Lane H 扩展它：加 `load_failures`（来自 `mm._load_failures`，dict[model_id, error]）、加 `runners`（来自 `app.state.runner_supervisors` —— Lane C 的 `RunnerSupervisor` 列表，每个报 `group_id / running / restart_count / pid`）。有 `_load_failures` 或有 runner 不 running → `status: "degraded"`。`app.state.runner_supervisors` 在 Lane H 落地时可能还没接入（Lane C 只造了 `RunnerSupervisor` 类，还没在 `main.py` 实例化）—— `/health` 用 `getattr(app.state, "runner_supervisors", [])` 兜底，supervisor 列表为空时 `runners: []`，不报错。

4. **F2 真 GPU-free 探针。** Lane C 的 `supervisor.py` 有 `_default_gpu_free_probe(gpus) -> bool` —— 当前是 `return True` 骨架。Lane H 在新模块 `src/runner/gpu_free_probe.py` 里实现真探针 `make_gpu_free_probe(baseline_free_mb)`：用 `gpu_monitor.poll_gpu_stats()` 查这些 GPU 的 `free_mb`，全部 ≥ baseline → True。`baseline_free_mb` 是「该 group 空载时应有的 free 显存下限」—— Lane H 取保守值（每卡 total 的 80%，从 `poll_gpu_stats` 的 `total_mb` 算）。`CUDA_VISIBLE_DEVICES=""` 测试环境下 `poll_gpu_stats()` 返回空列表 → 探针保守返回 True（不阻塞，与 Lane C 骨架行为一致）。`RunnerSupervisor` 的构造方默认值改为这个真探针（Lane C 已留 `gpu_free_probe` 注入参数，Lane H 只换默认实现 + 在 `main.py` 接入时显式传）。

**Tech Stack:** Python 3.12 / `pydantic`（`ModelSpec` 加字段）/ `asyncio`（`preload_residents` 是 `async`）/ FastAPI（`/health` 路由）/ `subprocess` 经 `gpu_monitor.poll_gpu_stats()`（nvidia-smi）/ pytest（`asyncio_mode = "auto"`）。复用 `services/model_manager.py` 已有的 `_load_failures` / `load_model` / `loaded_model_ids`，复用 Lane C 的 `src/runner/supervisor.py`。

> **注意 — 与 spec / 简报的偏差和歧义（已核实，须知会）：**
>
> 1. **spec §3.2 的 models.yaml 草图用 `local_path` + `vram_gb`，实际 `configs/models.yaml` 用 `paths.main` + `vram_mb`。** spec §3.2 是设计示意，不是真实文件结构。已读真实 `configs/models.yaml`：条目形如 `{id, type, adapter, paths: {main}, vram_mb, resident}`。Lane H 的 `preload_order` 按真实结构加在条目顶层（与 `resident` 同级），registry `_load` 用 `entry.get("preload_order")` 读。已在 Self-Review 标注。
>
> 2. **当前 `configs/models.yaml` 没有任何 `resident: true` 模型。** 全部 13 个条目都是 `resident: false`（已 grep 确认）。所以 Lane H 的 `preload_residents()` 在当前生产配置下是「遍历空集合，立即返回」—— 不是死代码，是「为未来 resident 模型准备好的、当前空转的正确路径」。测试用 fixture 注入 `resident: true` + `preload_order` 的 spec 来覆盖排序 / fail-soft 逻辑。已在 Self-Review 标注。
>
> 3. **`main.py` 现有 `_preload_image_model` 只 preload image 类。** spec §4.2 生命周期图说「按 preload_order 升序遍历 group_hint 匹配的 resident:true 模型」—— 跨 type。Lane H 的 `preload_residents()` 跨 type 遍历（image / tts，不含 llm —— llm 的 preload 是 Lane E 的 vLLM spawn，spec §4.2 明确 LLM 走另一条路）。`main.py` 改调 `preload_residents()` 后，原 `_preload_image_model` 的「成功后 invalidate cache + 推 ws 事件」副作用保留（作为 `preload_residents` 的可选 `on_loaded` 回调传入），不丢失现有行为。已在 Task 3 设计说明展开。
>
> 4. **`group_hint` / `group_require` 字段本 Lane 不实现。** spec §3.2 models.yaml 草图有 `group_hint` / `group_require`，但那是 Lane A（GPUAllocator NVLink-aware）的活。Lane H 的 `preload_residents()` 不按 group 过滤 —— 它遍历**所有** resident 模型。Lane A 落地后由后续 Lane 把 group 过滤接进来。本 Lane 只做 `preload_order` 排序。已在 Self-Review 标注。
>
> 5. **`app.state.runner_supervisors` 本 Lane 不负责创建。** Lane C 造了 `RunnerSupervisor` 类但没在 `main.py` 实例化（Lane C 的 Self-Review 明说「由后续 Lane H / scheduler 把 hardware.yaml 解析结果喂进 RunnerSupervisor 构造」）。Lane H 的 `/health` **读** `app.state.runner_supervisors`（用 `getattr` 兜底空列表），但**不创建** supervisor —— 真正的 supervisor 实例化 + 接 `hardware.yaml` 是 Lane A/scheduler 整合时做。本 Lane 把 `/health` 写成「supervisor 列表存在就报，不存在就 `runners: []`」，这样 Lane H 不被阻塞，且 Lane A 接入后 `/health` 自动开始报 runner 状态。已在 Self-Review 标注。
>
> 6. **`RunnerSupervisor` 没有公开的「health 快照」方法。** Lane C 的 `supervisor.py` 有 `is_running` property / `pid` property / `restart_count` 属性 / `group_id` / `gpus`，但没有一个聚合的 `health_snapshot()`。Lane H 给 `RunnerSupervisor` 加一个 `health_snapshot() -> dict` 方法（纯读现有属性，无副作用），`/health` 调它。这是对 Lane C 类的小幅扩展，不改其行为。已在 Task 4 设计说明展开。

---

## File Structure

| 文件 | Lane H 动作 | 责任 |
|---|---|---|
| `backend/src/services/inference/registry.py` | **修改** | `ModelSpec` 加 `preload_order: int | None = None`；`_load` / `add_from_scan` 从 yaml 读 |
| `backend/src/services/model_manager.py` | **修改** | 新增 `preload_residents(on_loaded=None)`：按 `preload_order` 升序 fail-soft preload resident 模型 |
| `backend/src/runner/gpu_free_probe.py` | **新建** | `make_gpu_free_probe(baseline_free_mb)` —— nvidia-smi-backed F2 探针工厂；无 GPU 保守返回 True |
| `backend/src/runner/supervisor.py` | **修改** | `_default_gpu_free_probe` 改用真探针；`RunnerSupervisor` 加 `health_snapshot() -> dict` |
| `backend/src/api/main.py` | **修改** | lifespan：`_preload_image_model` 段改调 `model_mgr.preload_residents(...)`；`/health` 路由扩展 `load_failures` + `runners` |
| `backend/configs/models.yaml` | **修改** | 字段说明注释（当前无 resident 模型 → 仅注释，未来 resident 条目加 `preload_order`） |
| `backend/tests/test_registry_preload_order.py` | **新建** | `ModelSpec.preload_order` 解析：yaml 有值 / 无值默认 None |
| `backend/tests/test_model_manager_preload_residents.py` | **新建** | `preload_residents`：升序、None 排最后、单个失败 fail-soft 不阻断、`_load_failures` 记录、`on_loaded` 回调 |
| `backend/tests/test_gpu_free_probe.py` | **新建** | `make_gpu_free_probe`：显存够 → True、不够 → False、无 GPU（空 stats）→ True |
| `backend/tests/test_health_endpoint.py` | **新建** | `/health` 扩展：含 `load_failures` / `runners`；有 failure → `status: degraded`；无 supervisor → `runners: []` |

> 测试基础设施复用：`tests/conftest.py` 强制 `ADMIN_PASSWORD=""`（admin gate 关）+ `NOUS_DISABLE_BG_TASKS=1`（lifespan 背景 task 不起，含 preload）+ `NOUS_DISABLE_FRONTEND_MOUNT=1`（SPA catch-all 不挂，新路由测试需要）+ `CUDA_VISIBLE_DEVICES=""`（torch / nvidia-smi 看不到 GPU）。`/health` 测试用 conftest 既有的 async client fixture。

---

## Task 1: `ModelSpec` 加 `preload_order` 字段

spec §3.2 models.yaml 草图列了 `preload_order: int`。`ModelSpec`（`registry.py:12`）现有 `resident: bool` 但没有 `preload_order`。先把字段加上 —— 后面 `preload_residents` 的排序依赖它。

**Files:**
- Modify: `backend/src/services/inference/registry.py`
- Test: `backend/tests/test_registry_preload_order.py`（新建）

- [ ] **Step 1: 写失败测试 — preload_order 从 yaml 解析**

新建 `backend/tests/test_registry_preload_order.py`：
```python
"""Lane H: ModelSpec.preload_order 字段解析测试。"""
import textwrap
from pathlib import Path

from src.services.inference.registry import ModelRegistry, ModelSpec


def _write_yaml(tmp_path: Path, body: str) -> str:
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_preload_order_defaults_to_none():
    """yaml 条目没写 preload_order → spec.preload_order 是 None。"""
    spec = ModelSpec(
        id="m", model_type="image", adapter_class="fake",
        paths={"main": "/fake"}, vram_mb=1024,
    )
    assert spec.preload_order is None


def test_preload_order_read_from_yaml(tmp_path):
    """yaml 条目写了 preload_order → registry 读进 spec。"""
    yaml_path = _write_yaml(tmp_path, """
        models:
          - id: early
            type: image
            adapter: fake.Adapter
            paths: {main: /fake/early}
            vram_mb: 1024
            resident: true
            preload_order: 10
          - id: late
            type: tts
            adapter: fake.Adapter
            paths: {main: /fake/late}
            vram_mb: 512
            resident: true
            preload_order: 20
          - id: unordered
            type: image
            adapter: fake.Adapter
            paths: {main: /fake/unordered}
            vram_mb: 256
            resident: true
    """)
    reg = ModelRegistry(yaml_path)
    assert reg.get("early").preload_order == 10
    assert reg.get("late").preload_order == 20
    assert reg.get("unordered").preload_order is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_registry_preload_order.py -v`
Expected: `test_preload_order_defaults_to_none` FAIL（`ModelSpec` 无 `preload_order` 属性 → `AttributeError`），`test_preload_order_read_from_yaml` FAIL（同因）。

- [ ] **Step 3: 给 `ModelSpec` 加字段 + registry 读取**

`backend/src/services/inference/registry.py`，`ModelSpec` 类 `gpu` 字段下方加一行：
```python
    gpu: int | list[int] | None = None
    preload_order: int | None = None
```

`_load` 方法的 `ModelSpec(...)` 构造里，`gpu=entry.get("gpu"),` 下方加：
```python
                gpu=entry.get("gpu"),
                preload_order=entry.get("preload_order"),
```

`add_from_scan` 方法的 `ModelSpec(...)` 构造里，同样在 `gpu=cfg.get("gpu"),` 下方加：
```python
            gpu=cfg.get("gpu"),
            preload_order=cfg.get("preload_order"),
```
（scanner 合成的 spec 一般不带 `preload_order`，`cfg.get` 返回 `None` —— 与字段默认一致，安全。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_registry_preload_order.py -v`
Expected: 2 个用例全 PASS。

- [ ] **Step 5: 跑 registry 既有 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "registry or model_manager"`
Expected: PASS。`preload_order` 是带默认值的纯新增字段，不动既有路径。

- [ ] **Step 6: lint 预检 + Commit**

```bash
cd backend && ruff check src/services/inference/registry.py tests/test_registry_preload_order.py
git add src/services/inference/registry.py tests/test_registry_preload_order.py
git commit -m "feat(registry): add ModelSpec.preload_order field (spec 3.2)

ModelSpec had resident:bool but not the preload_order:int the spec 3.2
models.yaml sketch lists. Lane H adds preload_order (defaults None),
read from yaml in both _load and add_from_scan. Lane H resident-preload
ordering builds on this. V1.5 Lane H."
```

---

## Task 2: `ModelManager.preload_residents()` —— 按 preload_order 升序 fail-soft

spec §4.2 生命周期图：「按 preload_order 升序遍历 resident:true 模型；load_failed 写 `_load_failures[model_key]`；不阻断后续 preload，不阻断 API server start」。spec §5.2 测试要点：「resident:true + preload_order 升序；null 排最后」。当前 `main.py` 的 `_preload_image_model` 是无序的、只 image 类。Lane H 把「排序 + 跨 type fail-soft 遍历」收进 `ModelManager.preload_residents()`。

**Files:**
- Modify: `backend/src/services/model_manager.py`
- Test: `backend/tests/test_model_manager_preload_residents.py`（新建）

- [ ] **Step 1: 写失败测试 — preload_residents 排序 + fail-soft**

新建 `backend/tests/test_model_manager_preload_residents.py`：
```python
"""Lane H: ModelManager.preload_residents 测试 —— preload_order 升序 + fail-soft。"""
import pytest
from unittest.mock import MagicMock

from src.services.inference.registry import ModelSpec
from src.services.model_manager import ModelManager


def _spec(model_id, *, resident=True, preload_order=None, model_type="image"):
    return ModelSpec(
        id=model_id, model_type=model_type, adapter_class="fake",
        paths={"main": f"/fake/{model_id}"}, vram_mb=1024,
        resident=resident, preload_order=preload_order,
    )


def _make_manager(specs):
    registry = MagicMock()
    registry.specs = specs
    registry.get = lambda mid: next((s for s in specs if s.id == mid), None)
    registry.add_from_scan = MagicMock(return_value=None)
    allocator = MagicMock()
    return ModelManager(registry=registry, allocator=allocator)


@pytest.mark.asyncio
async def test_preload_residents_orders_by_preload_order():
    """resident 模型按 preload_order 升序 load；preload_order=None 排最后。"""
    specs = [
        _spec("late", preload_order=30),
        _spec("none-a"),                       # preload_order None
        _spec("early", preload_order=10),
        _spec("mid", preload_order=20),
        _spec("none-b"),                       # preload_order None
    ]
    mm = _make_manager(specs)
    load_order: list[str] = []

    async def _fake_load(model_id, **kw):
        load_order.append(model_id)

    mm.load_model = _fake_load
    await mm.preload_residents()
    # 有序的在前（10/20/30），None 的在最后（保持 registry FIFO）
    assert load_order[:3] == ["early", "mid", "late"]
    assert set(load_order[3:]) == {"none-a", "none-b"}


@pytest.mark.asyncio
async def test_preload_residents_skips_non_resident():
    """resident:false 的模型不 preload。"""
    specs = [
        _spec("res", preload_order=10, resident=True),
        _spec("transient", preload_order=5, resident=False),
    ]
    mm = _make_manager(specs)
    loaded: list[str] = []
    mm.load_model = lambda mid, **kw: loaded.append(mid)  # noqa
    # load_model 需是 async —— 用 async wrapper
    async def _fake_load(model_id, **kw):
        loaded.append(model_id)
    mm.load_model = _fake_load
    await mm.preload_residents()
    assert loaded == ["res"]


@pytest.mark.asyncio
async def test_preload_residents_fail_soft_does_not_block():
    """单个模型 load 失败 → 写 _load_failures + 继续下一个，不向上抛。"""
    specs = [
        _spec("ok-1", preload_order=10),
        _spec("boom", preload_order=20),
        _spec("ok-2", preload_order=30),
    ]
    mm = _make_manager(specs)
    loaded: list[str] = []

    async def _fake_load(model_id, **kw):
        if model_id == "boom":
            raise RuntimeError("CUDA out of memory (simulated)")
        loaded.append(model_id)

    mm.load_model = _fake_load
    # 不抛异常 —— fail-soft
    await mm.preload_residents()
    # boom 失败但 ok-1 / ok-2 仍 load 了
    assert loaded == ["ok-1", "ok-2"]
    # 失败记录进 _load_failures
    assert "boom" in mm._load_failures
    assert "out of memory" in mm._load_failures["boom"].lower()


@pytest.mark.asyncio
async def test_preload_residents_invokes_on_loaded_callback():
    """每个成功 load 的模型触发 on_loaded(model_id) 回调（main.py 用它 invalidate cache + 推 ws）。"""
    specs = [_spec("a", preload_order=10), _spec("b", preload_order=20)]
    mm = _make_manager(specs)

    async def _fake_load(model_id, **kw):
        pass

    mm.load_model = _fake_load
    notified: list[str] = []

    async def _on_loaded(model_id):
        notified.append(model_id)

    await mm.preload_residents(on_loaded=_on_loaded)
    assert notified == ["a", "b"]


@pytest.mark.asyncio
async def test_preload_residents_callback_failure_is_swallowed():
    """on_loaded 回调本身抛异常 → 不影响后续 preload（回调是 best-effort）。"""
    specs = [_spec("a", preload_order=10), _spec("b", preload_order=20)]
    mm = _make_manager(specs)

    async def _fake_load(model_id, **kw):
        pass

    mm.load_model = _fake_load

    async def _bad_callback(model_id):
        raise RuntimeError("ws broadcast failed")

    # 不抛 —— 回调失败被吞
    await mm.preload_residents(on_loaded=_bad_callback)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_preload_residents.py -v`
Expected: 5 个用例全 FAIL —— `AttributeError: 'ModelManager' object has no attribute 'preload_residents'`。

- [ ] **Step 3: 实现 `preload_residents`**

`backend/src/services/model_manager.py`，在 `check_idle_models` 方法**之后**（约 :458，`evict_lru` 之前）插入：
```python
    async def preload_residents(
        self,
        on_loaded: "Callable[[str], Awaitable[None]] | None" = None,
    ) -> None:
        """Startup preload of resident models, ordered by `preload_order` (spec 4.2).

        遍历 registry 里所有 `resident:true` 的 spec，按 `preload_order` 升序
        load。`preload_order` 为 None 的排在最后（保持 registry 的 FIFO 顺序）。

        **Fail-soft（spec 4.3）**：单个模型 `load_model` 抛任何异常 → 把原因写进
        `_load_failures[model_id]` + 继续下一个模型，**绝不向上抛**。这样某个
        resident 模型 OOM / 文件损坏不会阻断 API server 启动，也不会阻断后面
        其它 resident 模型的 preload。失败的模型在 `/health` 的 `load_failures`
        里可见，Dashboard 据此显示 degraded banner + Retry。

        Parameters
        ----------
        on_loaded:
            可选回调，每个模型成功 load 后以 `model_id` 调用一次。`main.py` 用它
            做「invalidate engines/models cache + 推 ws/models 事件」。回调本身
            抛异常会被吞掉（best-effort，不影响 preload 流程）。

        LLM 类模型不在此处 preload —— vLLM 的 spawn / health 是 LLM Runner 的
        职责（spec 4.2），走另一条路。本方法只处理 image / tts 等进 image/TTS
        runner 的 resident 模型。
        """
        residents = [s for s in self._registry.specs if s.resident and s.model_type != "llm"]
        # 升序 key：preload_order 有值的在前（按值升序），None 的统一排到最后。
        # (0, order) < (1, 0) 保证所有 None 都在所有有值的之后。
        residents.sort(
            key=lambda s: (1, 0) if s.preload_order is None else (0, s.preload_order)
        )
        if residents:
            logger.info(
                "Preloading %d resident model(s) in order: %s",
                len(residents), [s.id for s in residents],
            )
        for spec in residents:
            try:
                await self.load_model(spec.id)
            except Exception as e:  # noqa: BLE001 — fail-soft is the whole point
                detail = f"{type(e).__name__}: {e}"
                self._load_failures[spec.id] = detail
                logger.warning("Resident preload failed for %s: %s", spec.id, detail)
                continue
            logger.info("Resident preload succeeded: %s", spec.id)
            if on_loaded is not None:
                try:
                    await on_loaded(spec.id)
                except Exception:  # noqa: BLE001 — callback is best-effort
                    logger.exception("preload_residents on_loaded callback failed for %s", spec.id)
```

`model_manager.py` 文件顶部的 import 区，把 `typing` 的 import 补上 `Awaitable` / `Callable`（grep 现有 import 行，若已有 `from typing import ...` 则追加，否则新增一行）：
```python
from typing import Awaitable, Callable
```
（注意：方法签名里 `on_loaded` 的类型注解用了字符串形式 `"Callable[...]"`，所以即使 import 漏了也不会运行时炸 —— 但为类型检查正确，仍补上 import。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_preload_residents.py -v`
Expected: 5 个用例全 PASS。

- [ ] **Step 5: 跑 ModelManager 既有 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_v2.py -q`
Expected: PASS（`preload_residents` 是纯新增方法）。

- [ ] **Step 6: lint 预检 + Commit**

```bash
cd backend && ruff check src/services/model_manager.py tests/test_model_manager_preload_residents.py
git add src/services/model_manager.py tests/test_model_manager_preload_residents.py
git commit -m "feat(model-manager): add preload_residents with fail-soft ordering (spec 4.2/4.3)

preload_residents iterates resident:true specs ordered by preload_order
ascending (None last), loading each. A load failure records the reason
into _load_failures and continues to the next model — it never raises,
so one OOM/corrupt-weights model cannot block API server startup or the
rest of the preload sequence. on_loaded callback lets main.py invalidate
caches + push ws events per success. LLM models are excluded (vLLM
spawn is the LLM runner's job). V1.5 Lane H."
```

---

## Task 3: `main.py` lifespan 改调 `preload_residents`

当前 `main.py` lifespan（约 :269-308）有一段 `_preload_image_model` 背景 task：无序、只 image 类。Lane H 把它换成调 `model_mgr.preload_residents(...)`，把原「成功后 invalidate cache + 推 ws」副作用收进 `on_loaded` 回调，行为不丢失，但现在按 `preload_order` 排序且跨 type。

**Files:**
- Modify: `backend/src/api/main.py`

- [ ] **Step 1: 看当前 lifespan preload 段**

Run:
```bash
cd backend && grep -n "_preload_image_model\|_image_preload_tasks\|preload_residents\|image_specs" src/api/main.py
```
Expected: 看到 `_preload_image_model` async 函数（约 :277）、`image_specs` 列表推导（约 :295）、`app.state._image_preload_tasks`（约 :301）。无 `preload_residents`。

- [ ] **Step 2: 替换 preload 段**

`backend/src/api/main.py`，把 `_bg_tasks_disabled` 为 False 分支里、从 `# Image models marked resident: preload in the background`（约 :269）那段注释开始、到 `app.state._image_preload_tasks = [...]` + 紧跟的 `if image_specs:` 日志块结束（约 :308）为止的**整段**，替换为：
```python
        # Resident models marked resident: preload in the background, ordered
        # by preload_order ascending (spec 4.2). The ~120s diffusers compose
        # must not block /health (cloudflared / systemd probes would mark the
        # backend down). preload_residents is fail-soft: a single model's
        # OOM / corrupt-weights failure records into mm._load_failures and is
        # surfaced on /health — it never blocks startup or the rest of the
        # preload sequence (spec 4.3). on_loaded flips the engines/models
        # cache + UI badge within ~1s per successful load.
        async def _on_resident_loaded(spec_id: str) -> None:
            from src.api.response_cache import invalidate as _invalidate
            _invalidate("models", "engines")
            from src.api.websocket import ws_manager as _ws
            await _ws.broadcast_model_status(spec_id, "loaded")

        # Persist the task ref so 3.11+ doesn't garbage-collect a still-running
        # background coroutine and silently drop the preload.
        app.state._resident_preload_task = asyncio.create_task(
            model_mgr.preload_residents(on_loaded=_on_resident_loaded)
        )
```
（注意：原代码里 `reconnected` 集合用于跳过已 reconnect 的 vLLM —— 那是 LLM 类，`preload_residents` 本来就 `model_type != "llm"` 排除了 llm，所以无需再传 `reconnected`。原 `_preload_image_model` 的失败分支「写 `_load_failures` + 推 error ws 事件」—— 写 `_load_failures` 现在由 `preload_residents` 内部做了；error ws 事件本 Lane 暂不在 `on_loaded` 里补，因为 `on_loaded` 只在成功时调 —— 失败的 ws 通知留给 Lane I 的 Dashboard 轮询 `/health` 来体现。这是有意的简化：`/health` 的 `load_failures` 是失败态的单一真相源。）

- [ ] **Step 3: 跑 lifespan / startup 相关 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "smoke or startup or integration_smoke or api_errors"`
Expected: PASS。注意：conftest 设了 `NOUS_DISABLE_BG_TASKS=1`，所以测试里这段 preload 分支根本不执行 —— 测试通过只证明「没引入 import error / 语法错」。preload 的真实行为由 Task 2 的单测覆盖。

- [ ] **Step 4: 跑全 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。无 `NameError` / `AttributeError`（特别确认没有别处引用被删的 `_preload_image_model` / `_image_preload_tasks` / `image_specs`）。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/api/main.py
git add src/api/main.py
git commit -m "refactor(main): route resident preload through preload_residents

The lifespan _preload_image_model block was unordered and image-only.
It now calls model_mgr.preload_residents(on_loaded=...), which orders
by preload_order ascending and is fail-soft across model types. The
cache-invalidate + ws-broadcast side effects move into the on_loaded
callback so per-success UX is unchanged. Failure surfacing moves to
/health's load_failures (single source of truth). V1.5 Lane H."
```

---

## Task 4: F2 真 GPU-free 探针 + `RunnerSupervisor.health_snapshot`

Lane C 的 `supervisor.py:_default_gpu_free_probe` 是 `return True` 骨架。Lane H 实现真探针：用 `gpu_monitor.poll_gpu_stats()` 查目标 GPU 的 `free_mb`，全部回落到基线才放行（spec §4.2 F2：死进程 CUDA context 回收异步，纯 backoff 不保证 context 已清）。同时给 `RunnerSupervisor` 加 `health_snapshot()` 供 Task 5 的 `/health` 用。

**Files:**
- Create: `backend/src/runner/gpu_free_probe.py`
- Modify: `backend/src/runner/supervisor.py`
- Test: `backend/tests/test_gpu_free_probe.py`（新建）

- [ ] **Step 1: 写失败测试 — make_gpu_free_probe**

新建 `backend/tests/test_gpu_free_probe.py`：
```python
"""Lane H: F2 GPU-free 探针测试 —— nvidia-smi-backed，无 GPU 保守返回 True。"""
from src.runner.gpu_free_probe import make_gpu_free_probe


def test_probe_true_when_all_gpus_have_enough_free(monkeypatch):
    """目标 GPU 的 free_mb 全部 >= baseline → 探针返回 True。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
            {"index": 1, "free_mb": 23000, "total_mb": 24000, "used_mb": 1000,
             "utilization_pct": 2, "temperature": 38},
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is True


def test_probe_false_when_a_gpu_still_occupied(monkeypatch):
    """某个目标 GPU 的 free_mb < baseline（CUDA context 还没回收）→ False。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
            {"index": 1, "free_mb": 8000, "total_mb": 24000, "used_mb": 16000,
             "utilization_pct": 90, "temperature": 70},  # 还占着
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is False


def test_probe_true_when_no_gpu_stats(monkeypatch):
    """nvidia-smi 返回空（CUDA_VISIBLE_DEVICES='' 测试环境）→ 保守返回 True，不阻塞重启。"""
    monkeypatch.setattr("src.runner.gpu_free_probe.poll_gpu_stats", lambda: [])
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is True


def test_probe_true_when_target_gpu_missing_from_stats(monkeypatch):
    """目标 GPU index 不在 stats 里（拔卡 / 索引错位）→ 该 GPU 视为不可判定，
    保守返回 True 不卡死重启循环（宁可早重启也不无限等一个不存在的 GPU）。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 5]) is True  # GPU 5 不存在 → 不阻塞


def test_probe_default_baseline_from_total(monkeypatch):
    """不传 baseline_free_mb → 用每卡 total_mb 的 80% 作为基线。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 20000, "total_mb": 24000, "used_mb": 4000,
             "utilization_pct": 10, "temperature": 45},
        ],
    )
    # 24000 * 0.8 = 19200；free 20000 >= 19200 → True
    probe = make_gpu_free_probe()
    assert probe([0]) is True
    # 把 free 压到 19000 < 19200 → False
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 19000, "total_mb": 24000, "used_mb": 5000,
             "utilization_pct": 20, "temperature": 50},
        ],
    )
    probe2 = make_gpu_free_probe()
    assert probe2([0]) is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_free_probe.py -v`
Expected: 全 FAIL —— `ModuleNotFoundError: No module named 'src.runner.gpu_free_probe'`。

- [ ] **Step 3: 实现 `gpu_free_probe.py`**

新建 `backend/src/runner/gpu_free_probe.py`：
```python
"""F2 GPU-free 探针 —— runner 重启前确认 GPU 显存已回落（spec 4.2）。

spec 4.2 F2：runner crash（OOM / native fault）后，死进程的 CUDA context 回收
是**异步**的 —— nvidia-smi 可能要几秒才反映显存释放。RunnerSupervisor 的
RESTART_BACKOFF `[5,15,60,300]` 是「防 crash 风暴」的退避，**不保证** context
已清。所以重启前必须额外过这道 gate：轮询 nvidia-smi，直到该 group 的所有 GPU
的 free_mb 回落到基线，才 re-fork runner + re-preload resident 模型。

本模块产出注入给 RunnerSupervisor 的 `gpu_free_probe` 回调（Lane C 已留注入点，
`_default_gpu_free_probe` 是它的骨架，本模块换成真实现）。

无 GPU 环境（CUDA_VISIBLE_DEVICES='' 测试 / nvidia-smi 不可用）：poll_gpu_stats
返回空 → 探针保守返回 True，不阻塞重启（与 Lane C 骨架行为一致）。
"""
from __future__ import annotations

import logging
from typing import Callable

from src.services.gpu_monitor import poll_gpu_stats

logger = logging.getLogger(__name__)

# 默认基线：每卡 total 显存的 80%。空载的 GPU 应有 >=80% free；低于此
# 说明上一个进程的 CUDA context 还没回收干净。
_DEFAULT_BASELINE_FRACTION = 0.8


def make_gpu_free_probe(
    baseline_free_mb: int | None = None,
) -> Callable[[list[int]], bool]:
    """构造一个 GPU-free 探针 —— 传给 RunnerSupervisor 的 gpu_free_probe 参数。

    Parameters
    ----------
    baseline_free_mb:
        每张目标 GPU 的 free_mb 必须 >= 此值才算「free」。None → 对每张卡用
        `total_mb * 0.8` 动态算（不同显存的卡各用自己的基线）。

    Returns
    -------
    `probe(gpus: list[int]) -> bool`：传入该 group 的 GPU index 列表，全部回落
    到基线返回 True，否则 False。nvidia-smi 不可用 / 目标 GPU 缺失 → 保守 True
    （宁可早重启，也不无限等一个查不到状态的 GPU 把重启循环卡死）。
    """

    def _probe(gpus: list[int]) -> bool:
        stats = poll_gpu_stats()
        if not stats:
            # nvidia-smi 不可用 / 无 GPU —— 保守放行（不阻塞重启）。
            return True
        by_index = {s["index"]: s for s in stats}
        for gpu in gpus:
            gpu_stat = by_index.get(gpu)
            if gpu_stat is None:
                # 目标 GPU 不在 stats 里 —— 不可判定，保守放行。
                logger.warning(
                    "GPU-free probe: GPU %d not in nvidia-smi output, skipping gate",
                    gpu,
                )
                continue
            if baseline_free_mb is not None:
                threshold = baseline_free_mb
            else:
                threshold = int(gpu_stat["total_mb"] * _DEFAULT_BASELINE_FRACTION)
            if gpu_stat["free_mb"] < threshold:
                logger.info(
                    "GPU-free probe: GPU %d free=%dMB < baseline=%dMB, not yet free",
                    gpu, gpu_stat["free_mb"], threshold,
                )
                return False
        return True

    return _probe
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_free_probe.py -v`
Expected: 5 个用例全 PASS。

- [ ] **Step 5: 把真探针接进 `supervisor.py` + 加 `health_snapshot`**

`backend/src/runner/supervisor.py`，改两处：

(a) `_default_gpu_free_probe` 函数体改为委托真探针：
```python
def _default_gpu_free_probe(gpus: list[int]) -> bool:
    """生产用 GPU-free 探针：nvidia-smi 查这些 GPU 的显存是否回落到基线（spec 4.2 F2）。

    Lane H 把 Lane C 的 `return True` 骨架换成真实现 —— 委托
    `gpu_free_probe.make_gpu_free_probe()`（默认基线 = 每卡 total 的 80%）。
    无 GPU 环境下该探针仍保守返回 True，不阻塞重启。
    """
    from src.runner.gpu_free_probe import make_gpu_free_probe
    return make_gpu_free_probe()(gpus)
```
（`make_gpu_free_probe` 是局部 import —— 避免 `supervisor.py` 模块顶层 import `gpu_monitor` 链；Lane C 既有 import 区不动。）

(b) `RunnerSupervisor` 类里，`backoff_for` 方法**之后**加 `health_snapshot`：
```python
    def health_snapshot(self) -> dict:
        """给 /health 端点用的 runner 状态快照（spec 4.2 / Lane H）。

        纯读现有属性，无副作用。Lane I 的 Dashboard 用 `running` / `restart_count`
        判断是否显示 degraded banner + 「重启中 N/M」。
        """
        return {
            "group_id": self.group_id,
            "gpus": list(self.gpus),
            "running": self.is_running,
            "restart_count": self.restart_count,
            "pid": self.pid,
        }
```

- [ ] **Step 6: 跑 supervisor 既有 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_supervisor.py -q`
Expected: PASS。Lane C 的 supervisor 测试注入自己的 fake `gpu_free_probe`（`lambda gpus: True`），不走 `_default_gpu_free_probe`，所以换默认实现不影响它们。`health_snapshot` 是纯新增方法。

- [ ] **Step 7: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/gpu_free_probe.py src/runner/supervisor.py tests/test_gpu_free_probe.py
git add src/runner/gpu_free_probe.py src/runner/supervisor.py tests/test_gpu_free_probe.py
git commit -m "feat(runner): real nvidia-smi GPU-free probe for F2 gate (spec 4.2)

Lane C left _default_gpu_free_probe as a 'return True' skeleton. Lane H
implements make_gpu_free_probe: polls nvidia-smi, requires every target
GPU's free_mb back to baseline (default = 80% of total) before a runner
restart re-preloads — dead-process CUDA context teardown is async, so
RESTART_BACKOFF alone may be too short. No-GPU / missing-GPU cases
return True conservatively (never deadlock the restart loop). Also adds
RunnerSupervisor.health_snapshot() for the /health endpoint. V1.5 Lane H."
```

---

## Task 5: `/health` 扩展 —— `load_failures` + `runners`

`main.py` 的 `create_app()` 里 inline `/health`（约 :461-486）现在返回 `{status, database, gpus, models_loaded}`。Lane H 加 `load_failures`（来自 `mm._load_failures`）+ `runners`（来自 `app.state.runner_supervisors` 的 `health_snapshot()`）。有 failure 或有 runner 不 running → `status: "degraded"`。

**Files:**
- Modify: `backend/src/api/main.py`
- Test: `backend/tests/test_health_endpoint.py`（新建）

- [ ] **Step 1: 写失败测试 — /health 含 load_failures + runners**

新建 `backend/tests/test_health_endpoint.py`：
```python
"""Lane H: /health 端点扩展测试 —— load_failures + runners + degraded 状态。"""
import pytest

from src.api.main import create_app


@pytest.mark.asyncio
async def test_health_has_load_failures_and_runners_keys():
    """/health 返回体含 load_failures 和 runners 两个新字段。"""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "load_failures" in body
    assert "runners" in body
    assert isinstance(body["load_failures"], dict)
    assert isinstance(body["runners"], list)


@pytest.mark.asyncio
async def test_health_no_runners_when_supervisors_unset():
    """app.state.runner_supervisors 未设置（Lane A 还没接入）→ runners 是空列表，不报错。"""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["runners"] == []


@pytest.mark.asyncio
async def test_health_degraded_when_load_failure_present():
    """mm._load_failures 非空 → status 是 'degraded'，failure 内容出现在 load_failures。"""
    from httpx import ASGITransport, AsyncClient

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # lifespan 已跑完 → app.state.model_manager 存在。注入一个 load failure。
        mm = app.state.model_manager
        mm._load_failures["flux2-dev"] = "OutOfMemoryError: CUDA out of memory"
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["load_failures"]["flux2-dev"] == "OutOfMemoryError: CUDA out of memory"


@pytest.mark.asyncio
async def test_health_reports_runner_snapshot():
    """app.state.runner_supervisors 有 supervisor → runners 列表含其 health_snapshot。"""
    from httpx import ASGITransport, AsyncClient

    class _FakeSupervisor:
        def health_snapshot(self):
            return {
                "group_id": "image", "gpus": [2], "running": False,
                "restart_count": 2, "pid": None,
            }

    app = create_app()
    app.state.runner_supervisors = [_FakeSupervisor()]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runners"] == [{
        "group_id": "image", "gpus": [2], "running": False,
        "restart_count": 2, "pid": None,
    }]
    # runner 不 running → status degraded
    assert body["status"] == "degraded"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_health_endpoint.py -v`
Expected: 全 FAIL —— `/health` 当前返回体无 `load_failures` / `runners` key（`KeyError` / `assert ... in body` 失败），`test_health_degraded...` 也失败（当前 status 永远 "ok" 除非 DB 挂）。

- [ ] **Step 3: 扩展 `/health`**

`backend/src/api/main.py`，`create_app()` 里的 `health_check` handler（约 :461-486），把整个函数体替换为：
```python
    @app.get("/health")
    async def health_check():
        checks: dict = {"status": "ok"}

        # Check database
        try:
            from src.models.database import create_session_factory
            from sqlalchemy import text
            _sf = create_session_factory()
            async with _sf() as session:
                await session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
            checks["status"] = "degraded"

        # GPU availability
        from src.services.gpu_monitor import get_gpu_stats
        gpus = get_gpu_stats()
        checks["gpus"] = len(gpus)

        # Loaded models + resident-preload failures (spec 4.3). A non-empty
        # load_failures dict means at least one resident model failed to
        # preload — the Dashboard renders a degraded banner + Retry from this.
        mgr = getattr(app.state, "model_manager", None)
        checks["models_loaded"] = len(mgr.loaded_model_ids) if mgr else 0
        load_failures = dict(mgr._load_failures) if mgr else {}
        checks["load_failures"] = load_failures
        if load_failures:
            checks["status"] = "degraded"

        # Per-runner state (spec 4.2). runner_supervisors is populated by the
        # scheduler/Lane A integration; until then it's unset and runners is
        # []. A runner that isn't running (crashed / mid-restart) degrades.
        supervisors = getattr(app.state, "runner_supervisors", [])
        runners = [s.health_snapshot() for s in supervisors]
        checks["runners"] = runners
        if any(not r.get("running", False) for r in runners):
            checks["status"] = "degraded"

        return checks
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_health_endpoint.py -v`
Expected: 4 个用例全 PASS。

- [ ] **Step 5: 跑既有 /health 相关 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "cors or integration_smoke or api_errors"`
Expected: PASS。既有测试只断言 `/health` 返回 200 + 已有字段，新增字段不破坏它们。

- [ ] **Step 6: lint 预检 + Commit**

```bash
cd backend && ruff check src/api/main.py
git add src/api/main.py tests/test_health_endpoint.py
git commit -m "feat(health): surface load_failures + per-runner state on /health (spec 4.2/4.3)

/health now returns load_failures (mm._load_failures dict) and runners
(RunnerSupervisor.health_snapshot list). A non-empty load_failures or a
non-running runner sets status=degraded so the Dashboard can render a
degraded banner. runner_supervisors is read via getattr — unset until
Lane A wires it, runners is [] until then. V1.5 Lane H."
```

---

## Task 6: `models.yaml` 字段注释 + 整合验证

`configs/models.yaml` 当前无 resident 模型，所以 `preload_order` 暂时只需文档化。加注释说明字段语义，将来加 resident 模型时有据可依。

**Files:**
- Modify: `backend/configs/models.yaml`
- 验证：无新文件

- [ ] **Step 1: 给 `models.yaml` 加字段说明注释**

`backend/configs/models.yaml` 文件**顶部**（`models:` 行之前）加注释块：
```yaml
# Model registry. Each entry: {id, type, adapter, paths, vram_mb, ...}.
#
# resident: bool        — true → never auto-unloaded by the idle checker;
#                         preloaded at startup (see preload_order).
# preload_order: int    — startup preload priority, ASCENDING (lower loads
#                         first). Only meaningful when resident: true.
#                         Omit → preloaded last, after all ordered entries.
#                         preload is fail-soft: one model's failure records
#                         into _load_failures (surfaced on /health) and does
#                         NOT block API server startup. See spec 4.2/4.3.
```
（当前所有条目都 `resident: false` → 无条目需要加 `preload_order`。这一步纯文档。）

- [ ] **Step 2: 确认 yaml 仍合法 + registry 能加载**

Run:
```bash
cd backend && python -c "from src.services.inference.registry import ModelRegistry; r = ModelRegistry('configs/models.yaml'); print(f'{len(r.specs)} specs loaded'); print('residents:', [s.id for s in r.specs if s.resident])"
```
Expected: 打印 `13 specs loaded`（或当前实际条目数）、`residents: []`（当前无 resident 模型）。无 yaml 解析错。

- [ ] **Step 3: 全 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS，无 collection error。

- [ ] **Step 4: lint 全量预检**

Run: `cd backend && ruff check src/ tests/`
Expected: 无新增 lint 错误。

- [ ] **Step 5: 关键路径冒烟 —— /health 扩展字段**

启动后端（依项目方式），然后：
```bash
curl -s localhost:8000/health | python -m json.tool
```
Expected: 200，返回体含 `status` / `database` / `gpus` / `models_loaded` / `load_failures`（空 dict `{}`）/ `runners`（空 list `[]`）。无 500、无 traceback。`load_failures` 为空 + 无 runner → `status: "ok"`。

- [ ] **Step 6: Commit**

```bash
cd backend && git add configs/models.yaml
git commit -m "docs(models.yaml): document resident + preload_order field semantics

No resident models in the current config — this is field-semantics
documentation so future resident entries have a reference. V1.5 Lane H."
```

- [ ] **Step 7: 开 PR**

```bash
git push -u origin <lane-h-branch>
gh pr create --title "feat: V1.5 Lane H — resident preload_order + /health + F2 GPU-free gate" --body "$(cat <<'EOF'
## Summary
- `ModelSpec` 加 `preload_order: int | None` 字段，registry 从 yaml 读
- `ModelManager.preload_residents()`：按 `preload_order` 升序 fail-soft preload，单个失败写 `_load_failures` 不阻断启动
- `main.py` lifespan 改调 `preload_residents`，副作用收进 `on_loaded` 回调
- F2 真 GPU-free 探针（`gpu_free_probe.py`）：nvidia-smi-backed，替换 Lane C 的 `return True` 骨架
- `/health` 扩展：暴露 `load_failures` + `runners`（per-runner `health_snapshot`），degraded 状态判定

## Test plan
- [ ] `test_registry_preload_order.py` green（preload_order 解析）
- [ ] `test_model_manager_preload_residents.py` green（升序 / None 排最后 / fail-soft / on_loaded）
- [ ] `test_gpu_free_probe.py` green（显存够/不够/无 GPU/缺卡/默认基线）
- [ ] `test_health_endpoint.py` green（load_failures / runners / degraded / 无 supervisor）
- [ ] 全 suite green（pytest tests/）
- [ ] /health 冒烟：load_failures + runners 字段存在
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneH-resident-preload-health`。）

---

## Self-Review

**Spec 覆盖检查：** Lane H 在 spec「实施分 Lane」表里的职责是「resident `preload_order` + `_load_failures` + `/health` 扩展 + Runner 重启 GPU-free gate（F2）。依赖：D」。

- **resident `preload_order`**（spec §3.2 / §4.2）→ Task 1（`ModelSpec` 加字段）+ Task 2（`preload_residents` 升序遍历）+ Task 6（yaml 文档）。spec §5.2「`preload_order` 排序」测试要点「resident:true + preload_order 升序；null 排最后」→ `test_model_manager_preload_residents.py::test_preload_residents_orders_by_preload_order` 直接覆盖。
- **`_load_failures` fail-soft**（spec §4.2「load_failed 写 `_load_failures`，不阻断后续 preload，不阻断 API server start」+ §4.3）→ Task 2 的 `preload_residents` 内 `except Exception → 写 _load_failures + continue`，**绝不向上抛**。`test_preload_residents_fail_soft_does_not_block` 是简报点名的「关键测试」—— resident 模型 load 失败不阻断启动 + 失败在 `_load_failures` 可见（再经 Task 5 暴露到 `/health`）。
- **`/health` 扩展**（spec §4.2「暴露到 `/health`」+ §6.2 Dashboard degraded banner）→ Task 5。`/health` 加 `load_failures` + `runners`，degraded 状态判定。`test_health_degraded_when_load_failure_present` 闭合「fail-soft 失败 → `/health` 可见 → Dashboard degraded banner」这条链。
- **F2 GPU-free gate**（spec §4.2）→ Task 4。Lane C 的 `_default_gpu_free_probe` `return True` 骨架换成 `gpu_free_probe.make_gpu_free_probe()` 真探针（nvidia-smi-backed，基线 = 每卡 total 80%）。Lane C 的 `RunnerSupervisor._restart()` 第 4 步已经调 `self._gpu_free_probe(self.gpus)` 轮询 —— Lane H 只换探针实现，不动 supervisor 的 restart 流程。
- **依赖 D**：Lane D 给 `ModelManager` 加了 `get_or_load`（OOM-evict-retry）。Lane H 的 `preload_residents` 调的是 `load_model`（不是 `get_or_load`）—— 因为 preload 是「首次全新 load」，OOM 时不需要 evict 重试（启动期 GPU 是空的，evict 没东西可 evict）；fail-soft 直接记 `_load_failures` 即可。Lane H 不 import 任何 Lane D 新增的 symbol，「依赖 D」是逻辑顺序依赖（D 把 ModelManager 迁进 runner，H 在 runner 内的 ModelManager 上做 preload）而非代码依赖。**判断**：Lane H 的 Task 1/2/3/5（preload + /health）在主进程 ModelManager 上就能完整测试和落地，不被 Lane D 阻塞；Task 4（GPU-free 探针）只依赖 Lane C 的 `supervisor.py`。所以 Lane H 实际可在 Lane C 后、Lane D 前并行 —— 但保守起见仍按 spec 标「依赖 D」，执行时若 D 未 merge 也不会卡住（无代码耦合）。

**与 spec 的偏差（已在 plan 头部「注意」逐条列出，此处汇总）：**
1. spec §3.2 models.yaml 草图字段名（`local_path` / `vram_gb`）与真实文件（`paths.main` / `vram_mb`）不符 —— 按真实结构实现，`preload_order` 加在条目顶层。
2. 当前 `configs/models.yaml` 无 `resident: true` 模型 —— `preload_residents()` 在当前生产配置下空转，是「为未来准备好的正确路径」，测试用 fixture 注入 resident spec 覆盖逻辑。
3. `main.py` 原 `_preload_image_model` 只 image 类、无序 —— `preload_residents()` 跨 type（不含 llm）、按 `preload_order` 排序，原「成功后 invalidate cache + 推 ws」副作用收进 `on_loaded` 回调保留。
4. `group_hint` / `group_require`（spec §3.2 草图）本 Lane 不实现 —— 那是 Lane A（NVLink-aware allocator）的活，`preload_residents` 遍历所有 resident 模型不按 group 过滤。
5. `app.state.runner_supervisors` 本 Lane 不创建（Lane C 只造类、未实例化）—— `/health` 用 `getattr` 兜底空列表，Lane A/scheduler 接入后 `/health` 自动开始报 runner 状态。
6. `RunnerSupervisor` 无聚合 health 方法 —— Lane H 加 `health_snapshot()`（纯读现有属性，无副作用）。

**Spec 歧义 flag：** spec §4.2 的 GPU-free gate「显存回落到基线」没定义「基线」的具体数值。Lane H 取「每卡 total 显存的 80%」作为默认基线（`_DEFAULT_BASELINE_FRACTION = 0.8`），并让 `make_gpu_free_probe` 接受显式 `baseline_free_mb` 覆盖。理由：空载 GPU（仅驱动 + compositor 占用）通常有 >90% free，80% 留了余量避免误判，又足够低能检出「上个进程 context 还没回收」（OOM 进程往往占满显存）。这是判断不是 spec 明文 —— 已在 plan 头部 + 此处 flag，Lane A/J 整合时若发现误判可调 `baseline_free_mb`。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。6 个 Task 全是「写失败测试 → 跑确认失败 → 最小实现 → 跑确认通过 → lint → commit」闭环（Task 6 是文档 + 验证，无新测试但有冒烟）。所有代码（`ModelSpec` 字段、`preload_residents`、`gpu_free_probe.py`、`health_snapshot`、`/health` 扩展）完整给出，命令带预期输出。`_default_gpu_free_probe` 委托 `make_gpu_free_probe()` 不是 placeholder —— 是 Lane C 骨架的真实现替换。

**类型一致性：**
- `ModelSpec.preload_order: int | None = None` ↔ registry `_load` / `add_from_scan` 的 `entry.get("preload_order")` / `cfg.get("preload_order")`（缺失返回 `None`，与默认一致）。
- `preload_residents(on_loaded: Callable[[str], Awaitable[None]] | None)` ↔ `main.py` 传入的 `async def _on_resident_loaded(spec_id: str) -> None` 签名一致；测试的 `async def _on_loaded(model_id)` 一致。
- `make_gpu_free_probe(baseline_free_mb: int | None) -> Callable[[list[int]], bool]` ↔ `RunnerSupervisor.__init__` 的 `gpu_free_probe: Callable[[list[int]], bool]`（Lane C 已定义）签名一致；`_default_gpu_free_probe(gpus: list[int]) -> bool` 一致。
- `RunnerSupervisor.health_snapshot() -> dict` ↔ `/health` 的 `[s.health_snapshot() for s in supervisors]`，返回 dict 的 key（`group_id/gpus/running/restart_count/pid`）与 `test_health_reports_runner_snapshot` 断言一致。

**已知风险：**
1. **Lane C 必须先 merge。** Task 4 改 `src/runner/supervisor.py` + import `src/runner/gpu_free_probe.py` 进 supervisor —— 依赖 Lane C 的 `src/runner/` 目录存在。执行 Lane H 时若 Lane C 未 merge，Task 4 会 `ModuleNotFoundError`。Task 4 Step 5 改 supervisor 前应先 `ls src/runner/supervisor.py` 确认在位（若不在，先做 Lane C）。Task 1/2/3/5 不依赖 Lane C，可独立推进。
2. **`/health` 测试触发真 lifespan。** `test_health_endpoint.py` 用 `create_app()` + `AsyncClient` 会跑完整 lifespan。conftest 的 `NOUS_DISABLE_BG_TASKS=1` 保证 preload / memory_guard 等背景 task 不起，但 lifespan 仍会建 DB 表 / 扫 node 包 / 建 ModelManager —— 测试较慢（秒级）。这是既有 `/health` 测试（`test_cors.py` 等）就有的成本，Lane H 不引入新慢点。
3. **GPU-free gate 基线误判风险。** 见上方「Spec 歧义 flag」。80% 默认基线在「GPU 被其它非 nous 进程占用」时会误判为「还没 free」→ 重启循环多等几轮。缓解：`make_gpu_free_probe` 接受显式 `baseline_free_mb`，Lane A 接入 `hardware.yaml` 时可按每 group 实际情况传精确值。本机单 admin infra，GPU 一般只跑 nous，误判概率低。
