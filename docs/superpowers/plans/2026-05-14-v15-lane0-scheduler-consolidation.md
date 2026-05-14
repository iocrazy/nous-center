# V1.5 Lane 0: 调度器整合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把三个重叠的「模型加载状态」真相源收敛成一个（`services/model_manager.py`），删掉死代码与旧模块，为 V1.5 后续 Lane 清场。

**Architecture:** 经调用方审计确认：(1) `src/gpu/model_manager.py` + `src/gpu/vram_tracker.py` + `deps.py:get_model_manager()` 是死代码（零真实调用方）——直接删除；(2) `model_scheduler.py` 仅剩 2 个外部触点（`monitor.py` 读 loaded 列表、`gpu_monitor.py` 的 evict 逻辑），把这 2 个触点改道到 `services/model_manager.py` 已有的等价 API（`loaded_model_ids` / `evict_lru(gpu_index)`），然后删除整个 `model_scheduler.py`。零可观测行为变化。

**Tech Stack:** Python 3 / FastAPI / pytest / asyncio。`services/model_manager.py` 是保留的基础类（asyncio.Lock，`ModelManager(registry, allocator)`，存在 `app.state.model_manager`）。

> **⚠️ 与 spec 的偏差（已核实，须知会）：** V1.5 spec 的 Lane 0 描述（源自 plan-eng-review 的 G5）说要「设计并实现 `services/model_manager.py` + `src/gpu/model_manager.py` 的合并，后者的 VRAMTracker 是 NVLink allocator 需要的真实 VRAM 核算」。**这个判断基于未完整追踪调用方的假设。** 实际代码审计结论：`src/gpu/model_manager.py` 的 ModelManager 仅被 `deps.py:get_model_manager()` import，而该函数无任何真实调用方（grep 无 `Depends(get_model_manager)`、无 `from src.api.deps import get_model_manager`）；且 live 的 `GPUAllocator`（`gpu_allocator.py`）直接 poll nvidia-smi 取真实 free_mb，**不使用** VRAMTracker。所以没有「合并」要做，是「删除死代码」。Task 1 的审计会再次验证此结论——若审计发现实际有调用方，停下来重新评估，不要盲删。

---

## File Structure

| 文件 | Lane 0 动作 | 责任 |
|---|---|---|
| `backend/src/gpu/model_manager.py` | **删除** | 死代码：VRAMTracker 式 bookkeeping ModelManager，无真实调用方 |
| `backend/src/gpu/vram_tracker.py` | **删除** | 仅被上面那个死 ModelManager 使用 |
| `backend/src/api/deps.py` | **修改** | 删除 `get_model_manager()`（返回死的 gpu ModelManager）；保留 `get_storage()` |
| `backend/src/services/model_scheduler.py` | **删除** | 旧的模块全局调度器，被 `services/model_manager.py` 取代 |
| `backend/src/api/routes/monitor.py` | **修改** | `model_scheduler.get_status()["loaded"]` → `app.state.model_manager.loaded_model_ids` |
| `backend/src/services/gpu_monitor.py` | **修改** | `check_and_evict` 改道到 `model_manager.evict_lru(gpu_index)`；`memory_guard_loop` 接收 model_manager 实例 |
| `backend/src/api/main.py` | **修改** | `memory_guard_loop` 启动处把 `model_mgr` 传进去 |
| `backend/tests/test_model_manager.py` | **删除（审计确认后）** | 测的是死的 gpu ModelManager（Task 1 验证） |
| `backend/tests/test_model_scheduler.py` | **删除** | 测的是被删的 model_scheduler.py |
| `backend/tests/test_gpu_monitor_evict.py` | **新建** | check_and_evict 改道后的回归测试 |

---

## Task 1: 调用方审计（确认死代码边界）

不写测试，这是一次性的事实核查。Lane 0 的全部「删除」动作都依赖这个审计的结论。**若任何一步的实际输出与「Expected」不符，停下来报告，不要继续删除。**

**Files:** 无（只读审计）

- [ ] **Step 1: 审计 `src/gpu/model_manager.py` 的调用方**

Run:
```bash
cd backend && grep -rn "gpu.model_manager\|from src.gpu import model_manager" src/ tests/ --include="*.py"
```
Expected: 只有 `src/api/deps.py:5:from src.gpu.model_manager import ModelManager` 和 `tests/test_model_manager.py`（测试文件）。无其它生产代码引用。

- [ ] **Step 2: 审计 `deps.py:get_model_manager()` 的调用方**

Run:
```bash
cd backend && grep -rn "get_model_manager\|from src.api.deps import" src/ tests/ --include="*.py"
```
Expected: 只有 `deps.py` 自身的 `def get_model_manager`、以及 `from src.api.deps import get_storage` 这类对 **get_storage** 的引用。**没有任何** `Depends(get_model_manager)` 或 `import get_model_manager`。注意 `engines.py` 里的 `_get_model_manager(request)` 是局部 helper 读 `request.app.state.model_manager`（services 版），与 `deps.py:get_model_manager` 无关——确认它没 import deps 的版本。

- [ ] **Step 3: 审计 `vram_tracker.py` 的调用方**

Run:
```bash
cd backend && grep -rn "vram_tracker\|VRAMTracker" src/ tests/ --include="*.py"
```
Expected: 只有 `src/gpu/model_manager.py`（即将删除）和 `src/gpu/vram_tracker.py` 自身。无其它引用。

- [ ] **Step 4: 审计 `model_scheduler.py` 的外部调用方**

Run:
```bash
cd backend && grep -rn "model_scheduler" src/ tests/ --include="*.py" | grep -v "src/services/model_scheduler.py"
```
Expected: 生产代码里只有 `src/api/routes/monitor.py`（`get_status()["loaded"]`）和 `src/services/gpu_monitor.py`（`_lock` / `get_loaded_keys` / `_references` / `_last_used` / `unload_model`）。测试里有 `tests/test_model_scheduler.py`。**若 `main.py` 或 nodes/ 或别处也引用 model_scheduler，停下来报告**——说明它的改道面比预期大，需扩充 Task 5。

- [ ] **Step 5: 确认 `services/model_manager.py` 已有等价 API**

Run:
```bash
cd backend && grep -n "def loaded_model_ids\|def evict_lru\|def get_references\|last_used" src/services/model_manager.py
```
Expected: 看到 `loaded_model_ids` property、`evict_lru(self, gpu_index)`、`get_references`、`LoadedModel.last_used`。这些就是 monitor.py / gpu_monitor.py 改道的目标 API。

- [ ] **Step 6: 确认测试文件归属**

Run:
```bash
cd backend && head -15 tests/test_model_manager.py tests/test_model_manager_v2.py
```
Expected: `test_model_manager.py` import `from src.gpu.model_manager import ModelManager`（测死的 gpu 版，将删）；`test_model_manager_v2.py` import `services/model_manager`（测保留的版本，**不删**）。若归属相反，调整 Task 2 / Task 5 删除的测试文件名。

- [ ] **Step 7: 记录审计结论 + commit（仅文档）**

把 Step 1-6 的实际输出贴进本 plan 文件 Task 1 下方的「审计结论」小节（手动追加），或贴进 commit message。
```bash
git add docs/superpowers/plans/2026-05-14-v15-lane0-scheduler-consolidation.md
git commit -m "docs(plan): Lane 0 caller audit confirmed — gpu ModelManager + model_scheduler dead surface"
```
（若审计未改动 plan 文件，跳过本 commit。）

---

## Task 2: 删除死的 gpu ModelManager 栈

审计（Task 1 Step 1-3、6）已确认 `src/gpu/model_manager.py` + `vram_tracker.py` + `deps.py:get_model_manager` 无真实调用方。

**Files:**
- Delete: `backend/src/gpu/model_manager.py`
- Delete: `backend/src/gpu/vram_tracker.py`
- Delete: `backend/tests/test_model_manager.py`（Task 1 Step 6 确认其测的是 gpu 版）
- Modify: `backend/src/api/deps.py`

- [ ] **Step 1: 跑现有 suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS（记下通过数，作为删除后的对照基线）。

- [ ] **Step 2: 从 `deps.py` 删除 `get_model_manager`**

`backend/src/api/deps.py` 改为：
```python
from functools import lru_cache

from src.storage.nas import StorageService


@lru_cache
def get_storage() -> StorageService:
    return StorageService()
```
（删掉 `load_model_configs` / `get_gpus` / `ModelManager` 三个 import 和整个 `get_model_manager` 函数。）

- [ ] **Step 3: 删除两个 gpu 模块文件 + 其测试**

Run:
```bash
cd backend && git rm src/gpu/model_manager.py src/gpu/vram_tracker.py tests/test_model_manager.py
```

- [ ] **Step 4: 跑 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS，通过数 = Step 1 基线 减去 `test_model_manager.py` 里的用例数。无 import error、无 collection error。若出现 `ModuleNotFoundError: src.gpu.model_manager` 或 `src.gpu.vram_tracker`，说明 Task 1 审计漏了调用方——停下来补查。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/api/deps.py
git commit -m "refactor(gpu): remove dead VRAMTracker-based ModelManager stack

src/gpu/model_manager.py + vram_tracker.py + deps.py:get_model_manager
had zero real callers (deps.py:get_model_manager was never wired into
any Depends()). The live GPUAllocator polls nvidia-smi directly. Lane 0."
```

---

## Task 3: `monitor.py` 改道到 `model_manager.loaded_model_ids`

`monitor.py` 唯一用到 `model_scheduler` 的地方是读「已加载模型 key 列表」。`app.state.model_manager` 已有 `loaded_model_ids` property，行为等价。

**Files:**
- Modify: `backend/src/api/routes/monitor.py`（约 :193-198）
- Test: `backend/tests/test_api_monitor.py`

- [ ] **Step 1: 写失败测试 — monitor 端点不依赖 model_scheduler**

在 `backend/tests/test_api_monitor.py` 追加：
```python
def test_monitor_loaded_models_from_model_manager(client, monkeypatch):
    """monitor 端点的 loaded_models 来自 app.state.model_manager，不依赖 model_scheduler。"""
    import src.api.routes.monitor as monitor_mod

    # model_scheduler 模块被删后，monitor.py 不应再 import 它
    src = (monitor_mod.__file__)
    with open(src) as f:
        content = f.read()
    assert "model_scheduler" not in content, "monitor.py 不应再引用 model_scheduler"
```
（若 `client` fixture 名不同，对齐 `test_api_monitor.py` 现有 fixture。）

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_api_monitor.py::test_monitor_loaded_models_from_model_manager -v`
Expected: FAIL —— 当前 `monitor.py` 仍有 `from src.services import model_scheduler`。

- [ ] **Step 3: 改 `monitor.py` 改道到 model_manager**

`backend/src/api/routes/monitor.py` 约 :192-198，把：
```python
    # Add loaded models to GPU info
    from src.services import model_scheduler
    from src.config import load_model_configs
    from src.gpu.detector import get_device_for_engine

    configs = load_model_configs()
    loaded = model_scheduler.get_status()["loaded"]
```
改为：
```python
    # Add loaded models to GPU info
    from src.config import load_model_configs
    from src.gpu.detector import get_device_for_engine

    configs = load_model_configs()
    model_mgr = getattr(request.app.state, "model_manager", None)
    loaded = model_mgr.loaded_model_ids if model_mgr is not None else []
```
（`request` 在该 handler 作用域内已可用——同文件 :176 已用 `request.app.state`。下游 `for model_key in loaded:` 循环不变。）

- [ ] **Step 4: 跑测试确认通过 + 端点回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_api_monitor.py -v`
Expected: 新测试 PASS，`test_api_monitor.py` 原有用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/api/routes/monitor.py tests/test_api_monitor.py
git commit -m "refactor(monitor): read loaded models from model_manager, drop model_scheduler

monitor.py only used model_scheduler for the loaded-model-id list;
app.state.model_manager.loaded_model_ids is the equivalent. Lane 0."
```

---

## Task 4: `gpu_monitor.py` evict 逻辑改道到 `model_manager.evict_lru`

`gpu_monitor.py:check_and_evict` 现在手动伸进 `model_scheduler` 的私有状态（`_lock` / `_references` / `_last_used`）做 LRU 挑选 + force-unload。`services/model_manager.py` 已有 `evict_lru(gpu_index)`，内部已实现「跳过 resident、跳过 referenced、挑 last_used 最旧、force unload」——完全等价。`check_and_evict` / `memory_guard_loop` 需要拿到 model_manager 实例（它们是 main.py lifespan 启的后台 loop，不是 request-scoped）。

**Files:**
- Modify: `backend/src/services/gpu_monitor.py`（`check_and_evict` + `memory_guard_loop` 签名）
- Modify: `backend/src/api/main.py`（启动 `memory_guard_loop` 处传入 model_mgr）
- Test: `backend/tests/test_gpu_monitor_evict.py`（新建）

- [ ] **Step 1: 写失败测试 — check_and_evict 调 model_manager.evict_lru**

新建 `backend/tests/test_gpu_monitor_evict.py`：
```python
"""Lane 0: check_and_evict 改道到 model_manager.evict_lru 的回归测试。"""
import pytest

from src.services.gpu_monitor import check_and_evict


class _FakeModelManager:
    def __init__(self):
        self.evict_calls: list[int | None] = []

    def evict_lru(self, gpu_index=None):
        self.evict_calls.append(gpu_index)
        return None  # 没有可驱逐的模型


@pytest.mark.asyncio
async def test_check_and_evict_calls_model_manager_evict_lru(monkeypatch):
    """GPU 低显存时，check_and_evict 调 model_manager.evict_lru(该 GPU index)。"""
    # 伪造一个低显存 GPU
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_monitor_evict.py -v`
Expected: FAIL —— 当前 `check_and_evict` 签名是 `check_and_evict(reserved_gb=...)`，不接收 model_manager 参数，且内部用 `model_scheduler`。

- [ ] **Step 3: 重写 `gpu_monitor.py` 的 `check_and_evict` + `memory_guard_loop`**

`backend/src/services/gpu_monitor.py`，把 `check_and_evict` 和 `memory_guard_loop` 整体替换为：
```python
async def check_and_evict(model_manager, reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """检查 GPU 显存，低于阈值时让 model_manager 驱逐该 GPU 上的 LRU 模型。

    model_manager: services.model_manager.ModelManager 实例（evict_lru 内部已处理
    resident / referenced 跳过 + last_used 排序 + force unload）。
    """
    stats = poll_gpu_stats()
    for gpu in stats:
        free_gb = gpu["free_mb"] / 1024
        if free_gb < reserved_gb:
            logger.debug(
                "GPU %d low memory: %.1fGB free (threshold: %.1fGB). Evicting LRU...",
                gpu["index"], free_gb, reserved_gb,
            )
            evicted = model_manager.evict_lru(gpu_index=gpu["index"])
            if evicted:
                logger.info("Auto-evicted model %s from GPU %d", evicted, gpu["index"])


async def memory_guard_loop(model_manager, reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """后台 loop：每 POLL_INTERVAL_SECONDS 检查一次 GPU 显存。"""
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            await check_and_evict(model_manager, reserved_gb)
        except Exception as e:
            logger.warning("GPU memory guard failed: %s", e)
```
（删掉原 `check_and_evict` 里 `from src.services import model_scheduler` / `load_model_configs` / 手动 candidates 收集那一整段。`evict_lru` 是同步方法，不需要 `await`。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_gpu_monitor_evict.py -v`
Expected: 两个用例都 PASS。

- [ ] **Step 5: 改 `main.py` 启动 `memory_guard_loop` 处传入 model_mgr**

Run 先定位调用点：
```bash
cd backend && grep -n "memory_guard_loop\|model_mgr" src/api/main.py
```
在 `main.py` lifespan 里，`memory_guard_loop` 被 `asyncio.create_task(...)` 启动的那行（在 `model_mgr = ModelManager(registry=registry, allocator=allocator)` 之后）。把：
```python
asyncio.create_task(memory_guard_loop())
```
改为：
```python
asyncio.create_task(memory_guard_loop(model_mgr))
```
（若 main.py 用了 `reserved_gb` 实参，保留为第二个位置参数：`memory_guard_loop(model_mgr, reserved_gb=...)`。）

- [ ] **Step 6: 跑全 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。无 `model_scheduler` 相关 import error（注意：此时 model_scheduler.py 还在，Task 5 才删；但 gpu_monitor.py 已不再 import 它）。

- [ ] **Step 7: Commit**

```bash
cd backend && git add src/services/gpu_monitor.py src/api/main.py tests/test_gpu_monitor_evict.py
git commit -m "refactor(gpu_monitor): route eviction through model_manager.evict_lru

check_and_evict no longer reaches into model_scheduler private state;
it delegates to ModelManager.evict_lru(gpu_index), which already skips
resident/referenced models and picks LRU. memory_guard_loop now takes
the model_manager instance. Lane 0."
```

---

## Task 5: 删除 `model_scheduler.py`

Task 3、4 之后，`model_scheduler.py` 已无任何外部生产代码调用方（Task 1 Step 4 已确认外部触点只有 monitor.py + gpu_monitor.py，两者已改道）。

**Files:**
- Delete: `backend/src/services/model_scheduler.py`
- Delete: `backend/tests/test_model_scheduler.py`

- [ ] **Step 1: 复核无残留引用**

Run:
```bash
cd backend && grep -rn "model_scheduler" src/ --include="*.py"
```
Expected: **零输出**。若仍有命中，说明 Task 3/4 未改干净或 Task 1 审计漏了调用方——回去补，不要继续删。

- [ ] **Step 2: 删除模块 + 测试**

Run:
```bash
cd backend && git rm src/services/model_scheduler.py tests/test_model_scheduler.py
```

- [ ] **Step 3: 跑全 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。通过数 = Task 2 Step 4 基线 减去 `test_model_scheduler.py` 的用例数。无 `ModuleNotFoundError: src.services.model_scheduler`。

- [ ] **Step 4: Commit**

```bash
cd backend && git commit -m "refactor(scheduler): delete model_scheduler.py — superseded by model_manager

model_scheduler.py was the old module-global model lifecycle tracker.
Its only external touchpoints (monitor.py loaded-list, gpu_monitor.py
eviction) are rerouted to services/model_manager.py. get_llm_base_url
was dead code (zero callers) and goes with it — V1.5 Lane E will
establish the vLLM base-URL lookup fresh. Lane 0."
```

---

## Task 6: 整合验证 + 冒烟

**Files:** 无（验证）

- [ ] **Step 1: 全 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS，无 skip 异常、无 collection error。

- [ ] **Step 2: 确认三个真相源已收敛为一个**

Run:
```bash
cd backend && grep -rln "model_scheduler\|gpu.model_manager\|VRAMTracker" src/ --include="*.py"
```
Expected: **零输出**。全仓只剩 `services/model_manager.py` 一个模型生命周期真相源。

- [ ] **Step 3: lint 预检（push 前本地跑）**

Run: `cd backend && ruff check src/ tests/`
Expected: 无新增 lint 错误（删除文件可能消除一些原有告警，属正常）。

- [ ] **Step 4: 后端冒烟 — 启动 + monitor 端点**

启动后端（依项目方式，如 `python -m src.api.main` 或既有启动脚本），然后：
```bash
curl -s localhost:8000/api/v1/monitor/stats | python -m json.tool | head -30
```
Expected: 200，返回含 `gpus` 数组，每个 GPU 的 `loaded_models` 字段存在（可能为空数组，取决于当前是否有模型加载）。无 500、无 traceback。重点确认 `loaded_models` 不再因 model_scheduler 缺失而报错。

- [ ] **Step 5: Lane 0 收尾 commit（若有未提交的 plan 勾选）**

```bash
git add docs/superpowers/plans/2026-05-14-v15-lane0-scheduler-consolidation.md
git commit -m "docs(plan): Lane 0 complete — scheduler consolidation verified"
```

- [ ] **Step 6: 开 PR**

```bash
git push -u origin <lane-0-branch>
gh pr create --title "refactor: V1.5 Lane 0 — scheduler consolidation" --body "$(cat <<'EOF'
## Summary
- 删除死代码 `src/gpu/model_manager.py` + `vram_tracker.py` + `deps.py:get_model_manager`（零真实调用方）
- `monitor.py` / `gpu_monitor.py` 改道到 `services/model_manager.py` 的等价 API
- 删除 `model_scheduler.py`（旧模块全局调度器，已被取代；`get_llm_base_url` 死代码一并删）
- 全仓收敛为单一模型生命周期真相源

## Test plan
- [ ] 全 suite green（pytest tests/）
- [ ] monitor 端点冒烟：loaded_models 字段正常
- [ ] grep 确认无 model_scheduler / gpu.model_manager / VRAMTracker 残留
EOF
)"
```
（分支名按项目惯例，如 `refactor/v15-lane0-scheduler-consolidation`。）

---

## Self-Review

**Spec 覆盖检查：** Lane 0 在 spec §「实施分 Lane」表里的职责是「删 `model_scheduler.py`；`monitor.py` + `gpu_monitor.py` 改用 `model_manager`；删除前把 `get_llm_base_url()` 重新安置；设计并实现两个 ModelManager 的合并（G5）」。

- 删 `model_scheduler.py` → Task 5 ✓
- `monitor.py` 改道 → Task 3 ✓
- `gpu_monitor.py` 改道 → Task 4 ✓
- `get_llm_base_url` 重新安置 → **偏差**：审计发现它零调用方，是死代码，随 model_scheduler.py 删除（Task 5），不需要重新安置。V1.5 Lane E 届时新建 vLLM base-URL 查找。已在 plan 顶部「与 spec 的偏差」+ Task 5 commit message 明确说明。
- 两个 ModelManager 合并（G5）→ **偏差**：审计发现 `src/gpu/model_manager.py` 是死代码（零真实调用方），不是「互补」，是删除（Task 2）。已在 plan 顶部明确说明，Task 1 审计会再次验证；若审计推翻此结论则停下重评。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有删除动作前置审计任务（Task 1）+ 每个删除步骤都有「若实际输出与 Expected 不符则停」的护栏。测试代码完整给出。

**类型一致性：** `check_and_evict(model_manager, reserved_gb=...)` 与 `memory_guard_loop(model_manager, reserved_gb=...)` 签名一致；`memory_guard_loop` 内调 `check_and_evict(model_manager, reserved_gb)` 参数顺序一致；`evict_lru(gpu_index=...)` 与 `services/model_manager.py` 现有签名一致（Task 1 Step 5 验证）。

**已知风险：** Task 1 审计是整个 Lane 的前提。若审计发现 `src/gpu/model_manager.py` 或 `model_scheduler.py` 实际有未追踪到的调用方，Task 2/5 的删除范围需重评——每个删除任务都内置了「Expected 不符则停」护栏防止盲删。
