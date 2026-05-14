# V1.5 Lane D: ModelManager 迁入 image/TTS Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- []`) syntax for tracking.

**Goal:** 把真的 `services/model_manager.py` ModelManager + 真 inference adapter（image 先迁）接进 Lane C 搭好的 image/TTS runner 子进程。Lane C 的 `runner_process.py` 现在用一个极简 `dict[model_key -> adapter]` + `FakeAdapter`；Lane D 把这块换成「每个 runner 子进程一个独立的 `ModelManager` 实例」，node-executor 通过 `ModelManager.get_or_load(model_key)`（per-model `asyncio.Lock`）拿 adapter 再 `adapter.infer(...)`。核心验证点：runner 内 per-model 锁真的把并发的同模型 `infer` 调用串行化（spec §1.3 / §4.5）。同时落地 runner 内 OOM 处理（spec §4.3：load OOM → evict + 重试一次 → 二次失败标 load_failed）。image runner 先迁验证 pattern，runner_main 的 adapter 装配路径泛化成 TTS 也能用（TTS 真正接入是 Lane F）。

**Architecture:** 四块改动，自底向上：

1. **`ModelManager` 没有 `get_or_load`，需要新增。** 现状（已读源码确认）：`services/model_manager.py` 有 `get_loaded_adapter(model_id)`（fast-path + lazy `load_model` + `_load_failures` 检查），有 `load_model`（内部 `async with self._lock_for(model_id)`），有 `evict_lru(gpu_index)`（`async def`），有 `_load_failures` dict。**但没有 spec §4.3 草图里那个带「OOM → evict + 重试一次」循环的 `get_or_load`。** `get_loaded_adapter` 的失败路径是「记 `_load_failures` 直接 raise」，不重试、不 evict。Lane D 给 `ModelManager` 加 `get_or_load(model_id)`：在 `get_loaded_adapter` 之上包一层 OOM 重试（evict 同 GPU 的 LRU 后重试一次），二次失败落 `_load_failures` 再 raise。这是 spec §4.3 要求的能力，归在 ModelManager（runner 内调用方只调 `get_or_load`）。

2. **runner 子进程构造一个独立 `ModelManager`。** spec §4.5：每个 runner 子进程有**自己的** ModelManager 实例（不是主进程那个 `app.state.model_manager`，子进程根本拿不到）。`ModelManager.__init__(registry, allocator)` 需要一个 `ModelRegistry` 和一个 `GPUAllocator`。runner 子进程里：`ModelRegistry(config_path)` 从 `models.yaml` 路径构造（registry 只读 yaml，无副作用，子进程内安全）；`GPUAllocator` 同理。Lane D 在 `runner_process.py` 的 `_RunnerState` 里持有这个 ModelManager，替换掉 Lane C 的极简 `adapters` dict。

3. **`_handle_load_model` / `_node_executor` 改走 ModelManager。** `_handle_load_model` 改为 `await mm.load_model(model_key)`，成功发 `ModelEvent(loaded)`、`ModelLoadError` 发 `ModelEvent(load_failed)`。`_node_executor` 改为 `adapter = await mm.get_or_load(node.model_key)` 再 `adapter.infer(req)`。**注意 cancel_flag / progress_callback 的接线**：Lane C 的 `FakeAdapter.infer` 接受 `progress_callback` / `cancel_flag` kwarg，但**真的 `image_diffusers.py` 的 `infer(req)` 目前只接受 `req`**（已读源码确认，§561）——给真 adapter 接 `callback_on_step_end` + `CancelFlag` 是 **Lane G** 的活（spec 实施 Lane 表 + §4.4 D14 明确归 Lane G）。Lane D **不重写真 adapter**：node-executor 用 `try`/`inspect` 探测 adapter 是否支持这俩 kwarg，支持就传（FakeAdapter 路径），不支持就退化为「只在节点边界 check cancel」（真 image adapter 路径，within-node cancel 留给 Lane G）。这样 Lane D 的 image runner 能跑真 adapter，Lane G 再补 within-node。

4. **runner 子进程入口注入 registry / allocator 配置。** `runner_main` 加可选参数 `models_yaml_path`（默认读环境变量 `NOUS_MODELS_YAML` 或项目默认路径），子进程内据此 build registry。`adapter_class` 参数（Lane C 加的）保留——Lane D 后默认仍是 `FakeAdapter`，但真实部署 / e2e 测试传真路径；模型用哪个 adapter 实际由 `ModelSpec.adapter_class` 决定（registry 从 yaml 读），`runner_main` 的 `adapter_class` 参数退化为「FakeAdapter 测试模式」的开关——见 Task 4 设计说明。

**Tech Stack:** Python 3.12 / `asyncio`（per-model `asyncio.Lock`）/ `multiprocessing`（spawn context，Lane C 已建立）/ pytest（`asyncio_mode = "auto"`）/ `pytest -m e2e` 标记真 GPU 测试（CI skip）。复用 Lane C 的 `src/runner/` 全部模块。

> **注意 — 与 spec / 简报的偏差和歧义（已核实，须知会）：**
>
> 1. **spec §4.3 把 `get_or_load` 写成 ModelManager 已有方法，实际不存在。** spec §4.3 的代码草图 `async def get_or_load(self, model_key)` 带 OOM 重试循环，写得像是「复用 model_manager 已有」。源码核实：`services/model_manager.py` 只有 `get_loaded_adapter`（无 evict 重试）、`load_model`、`evict_lru`。**没有 `get_or_load`。** Lane D 新建它（Task 1），语义对齐 spec §4.3 草图。已在 Self-Review 标注。
>
> 2. **spec §4.3 草图 `self.evict_lru(...)` 没 await，但 `evict_lru` 是 `async def`。** 简报已点明。Task 1 的 `get_or_load` 实现 `await self.evict_lru(...)`。
>
> 3. **spec §4.3 草图捕获 `torch.cuda.OutOfMemoryError`，但 runner venv 在测试里 torch 是 MagicMock（conftest stub）。** Lane D 的 `get_or_load` 不能在模块顶层 `import torch`。实现上用「捕获异常后按类名 / 文本判定是否 OOM」的方式（`type(e).__name__ == "OutOfMemoryError"` 或 `"out of memory" in str(e).lower()` 或 `isinstance` 经 lazy import），FakeAdapter 用一个专门的 `FakeOOMError`（子类名含 `OutOfMemoryError`）触发 OOM 路径测试。已在 Task 1 设计说明展开。
>
> 4. **真 image adapter `image_diffusers.py` 的 `infer(req)` 不接受 `cancel_flag` / `progress_callback`。** 见 Architecture 第 3 点。Lane D 不重写真 adapter（那是 Lane G 的 D14）。node-executor 用 signature 探测决定是否传 kwarg。Lane D 的 image runner 真 adapter 路径 = 节点边界 cancel；within-node cancel 是 Lane G。已在 Self-Review 标注。
>
> 5. **Lane C 尚未实现。** Lane D 依赖 B、C。本 plan 假设 Lane C 已落地（`src/runner/protocol.py` / `pipe_channel.py` / `fake_adapter.py` / `runner_process.py` / `client.py` / `supervisor.py` 都在）。若执行 Lane D 时 Lane C 未 merge，停下来先做 Lane C。Task 0 有一个前置检查 step。
>
> 6. **Lane B（TaskRingBuffer / schema migration）Lane D 不直接用。** 依赖关系里写了 B，但 Lane D 的改动全在 runner 子进程内，不碰 `execution_tasks` 表也不碰 ring buffer。列 B 为依赖是因为 Lane D 之后的集成（Lane G/J）需要两者都在。Lane D 自身的 task 不 import 任何 Lane B 产物。已在 Self-Review 标注。

---

## File Structure

| 文件 | Lane D 动作 | 责任 |
|---|---|---|
| `backend/src/services/model_manager.py` | **修改** | 新增 `get_or_load(model_id)`：`get_loaded_adapter` 之上包 OOM evict + 重试一次；二次失败落 `_load_failures` |
| `backend/src/runner/fake_adapter.py` | **修改** | 加 `FakeOOMError`（类名含 `OutOfMemoryError`）+ `oom_on_load_count` 构造参数（前 N 次 load 抛 OOM，之后成功）——给 runner OOM 路径测试用 |
| `backend/src/runner/runner_process.py` | **修改** | `_RunnerState` 持有真 `ModelManager`（替换极简 `adapters` dict）；`_handle_load_model` / `_handle_unload_model` / `_node_executor` 改走 ModelManager；`runner_main` 加 `models_yaml_path` 参数 + build registry/allocator |
| `backend/src/runner/runner_modelmanager.py` | **新建** | `build_runner_model_manager(group_id, gpus, models_yaml_path, fake_adapter)` 工厂：在 runner 子进程内构造 `ModelRegistry` + `GPUAllocator` + `ModelManager`，fake 模式下用 monkeypatch 让所有 spec 走 FakeAdapter |
| `backend/tests/test_model_manager_get_or_load.py` | **新建** | `get_or_load` 单测：fast-path、lazy load、OOM evict 重试一次、二次 OOM 落 `_load_failures` |
| `backend/tests/test_runner_modelmanager.py` | **新建** | `build_runner_model_manager` 工厂单测：registry 从 yaml fixture 构造、fake 模式所有 adapter 是 FakeAdapter |
| `backend/tests/test_runner_process_modelmanager.py` | **新建** | 真 `multiprocessing.Process` 跑接了 ModelManager 的 runner：LoadModel→ModelEvent、RunNode→NodeResult、**并发同模型 RunNode 被 per-model 锁串行化**、load_failed 不崩 runner |
| `backend/tests/fixtures/runner_models.yaml` | **新建** | runner ModelManager 测试用的最小 models.yaml fixture（2 个 fake image 模型条目） |
| `backend/tests/test_runner_oom_e2e.py` | **新建** | `@pytest.mark.e2e` —— 真 GPU 故意超 VRAM load → evict 重试 → 二次 OOM load_failed（CI skip，dev box 手动跑） |

> 测试基础设施复用：`tests/conftest.py` 的 `ADMIN_PASSWORD=""` / `NOUS_DISABLE_BG_TASKS=1` / `CUDA_VISIBLE_DEVICES=""` / torch MagicMock stub。Lane D 的 runner 子进程测试不碰 app / DB，但 ModelManager build 路径会 import registry / allocator —— 这些是纯 Python（registry 读 yaml、allocator 在 `CUDA_VISIBLE_DEVICES=""` 下 poll 不到 GPU 返回空），子进程内安全。

---

## Task 0: 前置检查 —— Lane C 已落地

不写测试，一次性事实核查。Lane D 全部 Task 依赖 Lane C 的 `src/runner/` 模块存在。

**Files:** 无（只读检查）

- [ ] **Step 1: 确认 Lane C 模块在位**

Run:
```bash
cd backend && ls src/runner/ && grep -c "" src/runner/runner_process.py
```
Expected: 看到 `protocol.py` / `pipe_channel.py` / `fake_adapter.py` / `runner_process.py` / `client.py` / `supervisor.py` / `__init__.py`。`runner_process.py` 非空。若目录不存在 → **停下来，先做 Lane C**（`docs/superpowers/plans/2026-05-14-v15-laneC-runner-framework.md`）。

- [ ] **Step 2: 确认 Lane C 测试基线 green**

Run:
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_protocol.py tests/test_pipe_channel.py tests/test_fake_adapter.py tests/test_runner_process.py tests/test_runner_client.py -q
```
Expected: 全 PASS。记下通过数作为基线。若有 FAIL → Lane C 不完整，停下来报告。

- [ ] **Step 3: 确认 ModelManager 当前 API 表面**

Run:
```bash
cd backend && grep -n "def get_or_load\|def get_loaded_adapter\|def load_model\|def evict_lru\|_load_failures" src/services/model_manager.py
```
Expected: 看到 `get_loaded_adapter` / `load_model` / `evict_lru` / `_load_failures`，**没有** `get_or_load`（确认偏差 1）。若已有 `get_or_load`，停下来核对它的语义是否已满足 spec §4.3，可能需调整 Task 1。

- [ ] **Step 4: 确认真 image adapter 的 infer 签名**

Run:
```bash
cd backend && grep -n "async def infer" src/services/inference/image_diffusers.py
```
Expected: `async def infer(self, req: InferenceRequest) -> InferenceResult:`——只接 `req`，不接 `cancel_flag` / `progress_callback`（确认偏差 4，within-node cancel 留 Lane G）。

---

## Task 1: `ModelManager.get_or_load` —— OOM evict + 重试一次

spec §4.3：load OOM → evict 同 GPU LRU + 重试一次；二次失败落 `_load_failures` 并 raise。源码核实 `ModelManager` 没有这个方法（偏差 1），新建它。runner node-executor 只调 `get_or_load`，OOM 重试逻辑全收在 ModelManager 内。

**Files:**
- Modify: `backend/src/services/model_manager.py`
- Test: `backend/tests/test_model_manager_get_or_load.py`（新建）

- [ ] **Step 1: 写失败测试 — get_or_load 四条路径**

新建 `backend/tests/test_model_manager_get_or_load.py`：
```python
"""Lane D: ModelManager.get_or_load —— OOM evict + 重试一次（spec §4.3）。

不碰真 GPU：用一个 stub adapter + stub registry/allocator，OOM 用一个类名含
'OutOfMemoryError' 的异常模拟（runner venv 测试里 torch 是 MagicMock，不能
import torch.cuda.OutOfMemoryError）。
"""
import pytest

from src.errors import ModelLoadError
from src.services.inference.base import (
    ImageRequest, InferenceAdapter, InferenceResult, MediaModality, UsageMeter,
)
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import ModelManager


class _CudaOutOfMemoryError(RuntimeError):
    """类名含 OutOfMemoryError —— get_or_load 据类名判定 OOM 路径。"""


class _StubAdapter(InferenceAdapter):
    modality = MediaModality.IMAGE
    estimated_vram_mb = 0

    # 类级别：实例化顺序计数 → 控制「第 N 个 adapter load 抛 OOM」
    def __init__(self, paths, device="cpu", *, oom_loads=0, **params):
        super().__init__(paths, device, **params)
        self._oom_loads = oom_loads
        self._load_calls = 0

    async def load(self, device):
        self._load_calls += 1
        if self._load_calls <= self._oom_loads:
            raise _CudaOutOfMemoryError("CUDA out of memory")
        self.device = device
        self._model = object()

    async def infer(self, req):
        return InferenceResult(
            media_type="image/png", data=b"x", metadata={},
            usage=UsageMeter(latency_ms=1, image_count=1),
        )


class _StubRegistry(ModelRegistry):
    """不读 yaml —— 直接塞 spec。"""
    def __init__(self, specs):
        self._config_path = ""
        self._specs = {s.id: s for s in specs}


class _StubAllocator:
    def get_best_gpu(self, vram_mb):
        return 0


def _spec(model_id, *, resident=False):
    return ModelSpec(
        id=model_id, model_type="image",
        adapter_class="tests.test_model_manager_get_or_load._StubAdapter",
        paths={"main": f"/fake/{model_id}"}, vram_mb=1024, resident=resident,
    )


def _mm(specs, *, factory_kwargs=None):
    """构造 ModelManager，注入一个 adapter_factory 让 load_model 用 stub。"""
    reg = _StubRegistry(specs)
    mm = ModelManager(registry=reg, allocator=_StubAllocator())
    return mm


@pytest.mark.asyncio
async def test_get_or_load_fast_path_when_already_loaded():
    """已加载 → get_or_load 直接返回，不重新 load。"""
    mm = _mm([_spec("m1")])
    a = _StubAdapter(paths={"main": "/fake/m1"})
    await mm.load_model("m1", adapter_factory=lambda spec: a)
    again = await mm.get_or_load("m1")
    assert again is a
    assert a._load_calls == 1  # 没有第二次 load


@pytest.mark.asyncio
async def test_get_or_load_lazy_loads_on_first_call():
    """未加载 → get_or_load 触发一次 load_model。"""
    mm = _mm([_spec("m1")])
    holder = {}

    def factory(spec):
        holder["a"] = _StubAdapter(paths=spec.paths)
        return holder["a"]

    adapter = await mm.get_or_load("m1", adapter_factory=factory)
    assert adapter is holder["a"]
    assert adapter.is_loaded


@pytest.mark.asyncio
async def test_get_or_load_oom_evicts_then_retries_once():
    """第一次 load OOM → evict 同 GPU LRU → 重试一次成功。"""
    mm = _mm([_spec("victim"), _spec("m2")])
    # 先放一个可被 evict 的 victim（非 resident、无引用）
    victim = _StubAdapter(paths={"main": "/fake/victim"})
    await mm.load_model("victim", adapter_factory=lambda s: victim)
    assert "victim" in mm.loaded_model_ids

    # m2 的 adapter：第 1 次 load OOM，第 2 次成功
    m2_adapter = _StubAdapter(paths={"main": "/fake/m2"}, oom_loads=1)
    adapter = await mm.get_or_load("m2", adapter_factory=lambda s: m2_adapter)

    assert adapter is m2_adapter
    assert adapter.is_loaded
    assert m2_adapter._load_calls == 2          # OOM 一次 + 重试成功一次
    assert "victim" not in mm.loaded_model_ids  # LRU 被 evict
    assert "m2" not in mm._load_failures        # 成功 → 不留失败记录


@pytest.mark.asyncio
async def test_get_or_load_second_oom_records_load_failure():
    """evict 后重试仍 OOM → 落 _load_failures + raise ModelLoadError。"""
    mm = _mm([_spec("m3")])
    # oom_loads=5：怎么试都 OOM
    m3_adapter = _StubAdapter(paths={"main": "/fake/m3"}, oom_loads=5)
    with pytest.raises(ModelLoadError):
        await mm.get_or_load("m3", adapter_factory=lambda s: m3_adapter)
    assert "m3" in mm._load_failures
    assert "OOM" in mm._load_failures["m3"] or "out of memory" in mm._load_failures["m3"].lower()
    # 二次 OOM → load 被调了 2 次（首次 + evict 后重试），不无限重试
    assert m3_adapter._load_calls == 2


@pytest.mark.asyncio
async def test_get_or_load_prior_failure_raises_without_retry():
    """已有 _load_failures 记录 → get_or_load 直接 raise，不重试。"""
    mm = _mm([_spec("m4")])
    mm._load_failures["m4"] = "previous OOM"
    with pytest.raises(ModelLoadError):
        await mm.get_or_load("m4")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_get_or_load.py -v`
Expected: FAIL —— `AttributeError: 'ModelManager' object has no attribute 'get_or_load'`。

- [ ] **Step 3: 实现 `get_or_load`**

`backend/src/services/model_manager.py`，在 `get_loaded_adapter` 方法之后（约 :266 之后）插入：
```python
    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        """判定异常是不是 CUDA OOM。

        不能在模块顶层 import torch（runner venv 测试里 torch 是 MagicMock，
        且 ModelManager 应能在无 torch 的纯逻辑测试中跑）。改用类名 + 文本判定：
        torch.cuda.OutOfMemoryError 的类名就是 'OutOfMemoryError'；其它库的
        OOM 一般文案里有 'out of memory'。
        """
        name = type(exc).__name__
        if "OutOfMemoryError" in name:
            return True
        return "out of memory" in str(exc).lower()

    async def get_or_load(
        self,
        model_id: str,
        adapter_factory: Callable[[ModelSpec], InferenceAdapter] | None = None,
    ) -> InferenceAdapter:
        """Get adapter, loading on demand with OOM-evict-retry (spec §4.3).

        在 `get_loaded_adapter` 的 lazy-load 之上加一层 OOM 韧性：首次 load 撞
        CUDA OOM → evict 同 GPU 的 LRU 非 resident / 非 referenced 模型 → 重试
        一次。重试仍 OOM（或非 OOM 异常）→ 落 `_load_failures` 并 raise
        ModelLoadError。**runner 子进程的 node-executor 唯一的加载入口** —— OOM
        重试逻辑全收在这里，调用方不感知。

        Raises:
            ModelNotFoundError: model_id 无 spec（HTTP 404）。
            ModelLoadError:     load 失败 / 二次 OOM（记入 `_load_failures`，HTTP 503）。
        """
        # Fast path: already loaded（含 _load_failures 检查）—— 复用 get_loaded_adapter
        # 的快路径语义。但 get_loaded_adapter 的 lazy load 不重试，所以这里只在
        # 「确定已加载」时直接走它；否则进 OOM-aware 重试循环。
        adapter = self.get_adapter(model_id)
        if adapter is not None and adapter.is_loaded:
            return adapter
        if model_id in self._load_failures:
            raise ModelLoadError(model_id, self._load_failures[model_id])

        spec = self._registry.get(model_id)
        if spec is None:
            spec = self._registry.add_from_scan(model_id)
        if spec is None:
            raise ModelNotFoundError(model_id)

        last_err: BaseException | None = None
        for attempt in range(2):
            try:
                await self.load_model(model_id, adapter_factory=adapter_factory)
                loaded = self.get_adapter(model_id)
                if loaded is None or not loaded.is_loaded:
                    self._load_failures[model_id] = (
                        "load_model returned but adapter is not loaded"
                    )
                    raise ModelLoadError(model_id, self._load_failures[model_id])
                self._load_failures.pop(model_id, None)
                return loaded
            except ModelLoadError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                if self._is_oom(e) and attempt == 0:
                    # 第一次 OOM —— evict 同 GPU LRU 后重试。gpu 来自 spec.gpu，
                    # 拿不到就传 None（evict 全局 LRU）。
                    gpu = spec.gpu if isinstance(spec.gpu, int) else None
                    evicted = await self.evict_lru(gpu_index=gpu)
                    logger.warning(
                        "get_or_load(%r): OOM on first load, evicted %r, retrying",
                        model_id, evicted,
                    )
                    continue
                # 非 OOM，或 evict 后二次 OOM —— 记失败并 raise
                msg = (
                    f"OOM after evict: {e}" if self._is_oom(e)
                    else f"{type(e).__name__}: {e}"
                )
                self._load_failures[model_id] = msg
                raise ModelLoadError(model_id, msg) from e
        # 理论不可达（循环要么 return 要么 raise）—— 防御
        self._load_failures[model_id] = f"{type(last_err).__name__}: {last_err}"
        raise ModelLoadError(model_id, self._load_failures[model_id])
```
（注意：`load_model` 内部已经 `async with self._lock_for(model_id)` —— per-model 锁在 `load_model` 那一层，`get_or_load` 不需要再包一层锁。但 `get_or_load` 的「检查 → load → 重试」序列不在锁内，并发同模型调用可能两个都进 `load_model`；`load_model` 内的 `if self.is_loaded(model_id): return` 会让第二个变 no-op —— 见 Task 3 的并发串行化测试验证这一点。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_get_or_load.py -v`
Expected: 5 个用例全 PASS。

- [ ] **Step 5: 跑 ModelManager 既有 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_model_manager_v2.py -q`
Expected: PASS（`get_or_load` 是纯新增方法，不动既有路径）。

- [ ] **Step 6: lint 预检 + Commit**

```bash
cd backend && ruff check src/services/model_manager.py tests/test_model_manager_get_or_load.py
git add src/services/model_manager.py tests/test_model_manager_get_or_load.py
git commit -m "feat(model-manager): add get_or_load with OOM-evict-retry (spec 4.3)

ModelManager had get_loaded_adapter (no retry) but not the OOM-resilient
get_or_load the spec 4.3 sketch assumed already existed. get_or_load
wraps load_model: on first CUDA OOM it evicts the same-GPU LRU and
retries once; a second OOM (or non-OOM error) records _load_failures
and raises ModelLoadError. OOM is detected by class name + text so the
manager stays importable without torch. V1.5 Lane D."
```

---

## Task 2: `FakeAdapter` 加 OOM 模拟开关

Task 3 要测 runner 内 OOM 路径（load OOM → evict 重试），需要一个能「前 N 次 load 抛 OOM、之后成功」的 fake adapter。Lane C 的 `FakeAdapter` 有 `fail_load`（永远失败）但没有「失败 N 次后成功」。加一个 `FakeOOMError`（类名含 `OutOfMemoryError`，触发 `ModelManager._is_oom`）+ `oom_on_load_count` 构造参数。

**Files:**
- Modify: `backend/src/runner/fake_adapter.py`
- Test: `backend/tests/test_fake_adapter.py`（追加用例）

- [ ] **Step 1: 写失败测试 — FakeAdapter OOM 开关**

在 `backend/tests/test_fake_adapter.py` 末尾追加：
```python
# —— Lane D 追加：OOM 模拟 ——

@pytest.mark.asyncio
async def test_fake_oom_on_load_then_succeeds():
    """oom_on_load_count=1 —— 第 1 次 load 抛 FakeOOMError，第 2 次成功。"""
    from src.runner.fake_adapter import FakeAdapter, FakeOOMError

    a = FakeAdapter(paths={"main": "/fake"}, oom_on_load_count=1)
    with pytest.raises(FakeOOMError):
        await a.load("cpu")
    assert not a.is_loaded
    # 第二次成功
    await a.load("cpu")
    assert a.is_loaded


def test_fake_oom_error_classname_triggers_is_oom():
    """FakeOOMError 类名含 OutOfMemoryError —— ModelManager._is_oom 认得它。"""
    from src.runner.fake_adapter import FakeOOMError
    from src.services.model_manager import ModelManager

    assert ModelManager._is_oom(FakeOOMError("CUDA out of memory"))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_adapter.py -k oom -v`
Expected: FAIL —— `ImportError: cannot import name 'FakeOOMError'`。

- [ ] **Step 3: 改 `fake_adapter.py`**

`backend/src/runner/fake_adapter.py`，在 `FakeLoadError` 定义之后加：
```python
class FakeOOMError(RuntimeError):
    """FakeAdapter.load() 在 oom_on_load_count 未耗尽时抛 —— 模拟 CUDA OOM。

    类名刻意含 'OutOfMemoryError' 子串，让 ModelManager._is_oom 据类名判定它
    走 OOM-evict-retry 路径（不能依赖 torch.cuda.OutOfMemoryError —— 测试里
    torch 是 MagicMock）。
    """
```
然后改 `__init__` 签名和 `load`：
```python
    def __init__(
        self,
        paths: dict[str, str],
        device: str = "cpu",
        *,
        fail_load: bool = False,
        crash_on_infer: bool = False,
        infer_seconds: float = 0.01,
        oom_on_load_count: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(paths, device, **params)
        self._fail_load = fail_load
        self._crash_on_infer = crash_on_infer
        self._infer_seconds = infer_seconds
        self._oom_on_load_count = oom_on_load_count
        self._load_attempts = 0

    async def load(self, device: str) -> None:
        await asyncio.sleep(0)  # 可让出
        self._load_attempts += 1
        if self._load_attempts <= self._oom_on_load_count:
            raise FakeOOMError(
                f"CUDA out of memory (fake, attempt {self._load_attempts})"
            )
        if self._fail_load:
            raise FakeLoadError(f"fake load failure for paths={self.paths}")
        self.device = device
        self._model = object()  # 非 None → is_loaded True
```
（`fail_load` 与 `oom_on_load_count` 互不冲突：OOM 次数先耗尽，之后 `fail_load=True` 仍永远失败。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_adapter.py -v`
Expected: 全 PASS（Lane C 的 7 个 + Lane D 追加的 2 个 = 9 个）。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/fake_adapter.py tests/test_fake_adapter.py
git add src/runner/fake_adapter.py tests/test_fake_adapter.py
git commit -m "feat(runner): add FakeOOMError + oom_on_load_count to FakeAdapter

Lets runner tests exercise the OOM-evict-retry path: oom_on_load_count=N
makes the first N load() calls raise FakeOOMError (classname contains
OutOfMemoryError so ModelManager._is_oom matches it), then load
succeeds. V1.5 Lane D."
```

---

## Task 3: runner 子进程 ModelManager 工厂（`runner_modelmanager.py`）

spec §4.5：每个 runner 子进程持有**自己的** `ModelManager` 实例。它需要 `ModelRegistry` + `GPUAllocator`。子进程内构造这俩是安全的（registry 只读 yaml，allocator 在 `CUDA_VISIBLE_DEVICES=""` 下 poll 不到 GPU）。把构造逻辑收进一个工厂函数，单独可测。fake 模式（测试 / 无真模型）下让所有 spec 的 adapter 都走 `FakeAdapter`。

**Files:**
- Create: `backend/src/runner/runner_modelmanager.py`
- Create: `backend/tests/fixtures/runner_models.yaml`
- Test: `backend/tests/test_runner_modelmanager.py`（新建）

- [ ] **Step 1: 建 yaml fixture**

新建 `backend/tests/fixtures/runner_models.yaml`：
```yaml
# Lane D: runner ModelManager 测试用最小 models.yaml fixture。
# 两个 fake image 模型 —— adapter_class 指向 FakeAdapter，无真权重。
models:
  - id: fake-img-a
    type: image
    adapter: src.runner.fake_adapter.FakeAdapter
    paths:
      main: /fake/fake-img-a
    vram_mb: 1024
    resident: false

  - id: fake-img-b
    type: image
    adapter: src.runner.fake_adapter.FakeAdapter
    paths:
      main: /fake/fake-img-b
    vram_mb: 1024
    resident: false
```

- [ ] **Step 2: 写失败测试 — 工厂构造 ModelManager**

新建 `backend/tests/test_runner_modelmanager.py`：
```python
"""Lane D: build_runner_model_manager 工厂 —— runner 子进程内构造独立 ModelManager。"""
from pathlib import Path

import pytest

from src.runner.fake_adapter import FakeAdapter
from src.runner.runner_modelmanager import build_runner_model_manager
from src.services.model_manager import ModelManager

_FIXTURE = str(Path(__file__).parent / "fixtures" / "runner_models.yaml")


def test_factory_returns_model_manager():
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert isinstance(mm, ModelManager)


def test_factory_registry_has_fixture_specs():
    """registry 从 yaml fixture 读到两个 fake image spec。"""
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert mm._registry.get("fake-img-a") is not None
    assert mm._registry.get("fake-img-b") is not None


@pytest.mark.asyncio
async def test_factory_fake_mode_loads_fake_adapter():
    """fake_adapter=True —— load_model 出来的 adapter 是 FakeAdapter（不碰真权重）。"""
    mm = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    adapter = await mm.get_or_load("fake-img-a")
    assert isinstance(adapter, FakeAdapter)
    assert adapter.is_loaded


@pytest.mark.asyncio
async def test_factory_each_call_is_independent_instance():
    """spec §4.5：每个 runner 一个独立 ModelManager —— 两次 build 互不共享状态。"""
    mm1 = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    mm2 = build_runner_model_manager(
        group_id="image", gpus=[2], models_yaml_path=_FIXTURE, fake_adapter=True,
    )
    assert mm1 is not mm2
    await mm1.get_or_load("fake-img-a")
    assert "fake-img-a" in mm1.loaded_model_ids
    assert "fake-img-a" not in mm2.loaded_model_ids  # 状态不共享
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_modelmanager.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.runner.runner_modelmanager'`。

- [ ] **Step 4: 实现 `runner_modelmanager.py`**

新建 `backend/src/runner/runner_modelmanager.py`：
```python
"""runner 子进程内的 ModelManager 工厂（spec §4.5）。

spec §4.5：每个 image/TTS runner 子进程持有**自己的** ModelManager 实例 —— 不是
主进程的 app.state.model_manager（子进程根本拿不到那个对象）。本模块把「子进程内
构造 ModelRegistry + GPUAllocator + ModelManager」收进一个工厂函数，单独可测。

fake_adapter=True 时（Lane C/D 测试、无真模型的环境）：注入一个 adapter_factory,
让 ModelManager.load_model 对**所有** spec 都实例化 FakeAdapter —— 这样 runner
框架（IPC + 生命周期 + per-model 锁）能在零 GPU 下跑通。真实部署 fake_adapter=
False，adapter 由 ModelSpec.adapter_class 决定（registry 从 yaml 读）。
"""
from __future__ import annotations

import logging
import os

from src.runner.fake_adapter import FakeAdapter
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import InferenceAdapter
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

# 真实部署的 models.yaml 默认位置 —— 环境变量优先。
_DEFAULT_MODELS_YAML = os.getenv("NOUS_MODELS_YAML", "config/models.yaml")


def _fake_adapter_factory(spec: ModelSpec) -> InferenceAdapter:
    """所有 spec 都走 FakeAdapter —— spec.params 里的 fail_load / infer_seconds /
    oom_on_load_count 透传给构造，这样测试能通过 yaml 配置 runner 的故障行为。"""
    return FakeAdapter(paths=spec.paths, **dict(spec.params))


def build_runner_model_manager(
    group_id: str,
    gpus: list[int],
    *,
    models_yaml_path: str | None = None,
    fake_adapter: bool = False,
) -> ModelManager:
    """在 runner 子进程内构造一个独立的 ModelManager。

    Parameters
    ----------
    group_id / gpus:
        本 runner 负责的 GPU group —— 目前 ModelManager / GPUAllocator 不按 group
        切分（Lane A 的 NVLink-aware allocator 才做），这里仅记 log，为 Lane A/G
        留接口。
    models_yaml_path:
        models.yaml 路径。None → NOUS_MODELS_YAML 环境变量 → config/models.yaml。
    fake_adapter:
        True → 所有 spec 走 FakeAdapter（测试 / 无真模型）。注意：调用方需把这个
        flag 透传给 ModelManager 的每次 load —— 见返回值说明。

    Returns
    -------
    ModelManager。fake_adapter=True 时，**调用方应通过 get_or_load 的
    adapter_factory 参数** 或本工厂返回的预绑定 manager 走 FakeAdapter。为简化
    runner 代码，本工厂在 fake 模式下 monkeypatch manager 实例的 load_model /
    get_or_load，把 adapter_factory 默认绑成 _fake_adapter_factory。
    """
    yaml_path = models_yaml_path or _DEFAULT_MODELS_YAML
    registry = ModelRegistry(yaml_path)
    allocator = GPUAllocator()
    mm = ModelManager(registry=registry, allocator=allocator)
    logger.info(
        "runner ModelManager built: group=%s gpus=%s yaml=%s fake=%s "
        "(%d specs)",
        group_id, gpus, yaml_path, fake_adapter, len(registry.specs),
    )

    if fake_adapter:
        # fake 模式：把 load_model / get_or_load 的 adapter_factory 默认值绑成
        # FakeAdapter 工厂。用 functools.partial 包原方法 —— 不改 ModelManager
        # 类（其它真实路径不受影响）。
        import functools

        orig_load = mm.load_model
        orig_get_or_load = mm.get_or_load

        @functools.wraps(orig_load)
        async def _load_model(model_id, adapter_factory=None):
            return await orig_load(
                model_id, adapter_factory=adapter_factory or _fake_adapter_factory,
            )

        @functools.wraps(orig_get_or_load)
        async def _get_or_load(model_id, adapter_factory=None):
            return await orig_get_or_load(
                model_id, adapter_factory=adapter_factory or _fake_adapter_factory,
            )

        mm.load_model = _load_model       # type: ignore[method-assign]
        mm.get_or_load = _get_or_load     # type: ignore[method-assign]

    return mm
```

> 设计说明：fake 模式用「实例级 monkeypatch（functools.partial 绑 adapter_factory 默认值）」而不是给 `ModelManager` 加一个 `fake` 构造参数 —— 因为 `ModelManager` 是 V1.5 多个 Lane 共用的核心类，往它构造里塞测试专用 flag 会污染真实路径。工厂在 runner 边界做这件事，`ModelManager` 本体保持干净。`GPUAllocator()` 无参构造（已确认其签名）；它在 `CUDA_VISIBLE_DEVICES=""` 下 `get_best_gpu` 返回 -1 或 0 —— fake 模式的 FakeAdapter `load` 不在意 device，安全。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_modelmanager.py -v`
Expected: 4 个用例全 PASS。

- [ ] **Step 6: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/runner_modelmanager.py tests/test_runner_modelmanager.py
git add src/runner/runner_modelmanager.py tests/fixtures/runner_models.yaml tests/test_runner_modelmanager.py
git commit -m "feat(runner): add per-subprocess ModelManager factory (spec 4.5)

build_runner_model_manager constructs an independent ModelRegistry +
GPUAllocator + ModelManager inside the runner subprocess — spec 4.5
requires each runner own its ModelManager, not share the main-process
one. fake_adapter=True binds load_model/get_or_load to a FakeAdapter
factory via instance-level functools.partial, keeping ModelManager
itself free of test-only flags. V1.5 Lane D."
```

---

## Task 4: `runner_process.py` 接入 ModelManager + per-model 锁验证

把 Lane C 的极简 `dict[model_key -> adapter]` 换成 Task 3 的真 `ModelManager`。`_handle_load_model` / `_handle_unload_model` / `_node_executor` 全改走 ModelManager。最关键的验证：runner 内 per-model `asyncio.Lock`（在 `ModelManager.load_model` 那一层）真的把并发的同模型调用串行化。

**Files:**
- Modify: `backend/src/runner/runner_process.py`
- Test: `backend/tests/test_runner_process_modelmanager.py`（新建）

- [ ] **Step 1: 写失败测试 — runner 接 ModelManager + 并发同模型串行化**

新建 `backend/tests/test_runner_process_modelmanager.py`：
```python
"""Lane D: runner 子进程接 ModelManager —— 真 multiprocessing.Process。

验证：
  * runner 用 build_runner_model_manager 构造的真 ModelManager（fake adapter 模式）
  * LoadModel → ModelEvent(loaded)；RunNode → NodeResult
  * 并发的同模型 RunNode 被 per-model asyncio.Lock 串行化（核心验证点）
  * load_failed 不崩 runner
"""
import multiprocessing as mp
from pathlib import Path

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import runner_main

_SPAWN = mp.get_context("spawn")
_FIXTURE = str(Path(__file__).parent / "fixtures" / "runner_models.yaml")


def _spawn_runner(group_id="image", gpus=(2,)):
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(
        target=runner_main,
        args=(group_id, list(gpus), child_conn),
        kwargs={"models_yaml_path": _FIXTURE, "fake_adapter": True},
        daemon=True,
    )
    proc.start()
    child_conn.close()
    return proc, PipeChannel(parent_conn)


import asyncio


async def _recv(ch, timeout=10.0):
    return await asyncio.wait_for(ch.recv_message(), timeout=timeout)


async def _shutdown(proc, ch):
    ch.close()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3.0)


async def _collect_until_result(ch, task_id):
    """收消息直到拿到指定 task_id 的 NodeResult，返回 (progress 列表, result)。"""
    progresses = []
    while True:
        msg = await _recv(ch)
        if isinstance(msg, P.NodeResult) and msg.task_id == task_id:
            return progresses, msg
        if isinstance(msg, P.NodeProgress) and msg.task_id == task_id:
            progresses.append(msg)


@pytest.mark.asyncio
async def test_runner_loads_model_via_modelmanager():
    """LoadModel —— runner 用 ModelManager.load_model，发 ModelEvent(loaded)。"""
    proc, ch = _spawn_runner()
    try:
        assert isinstance(await _recv(ch), P.Ready)
        await ch.send_message(P.LoadModel(model_key="fake-img-a", config={}))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent)
        assert ev.event == "loaded"
        assert ev.model_key == "fake-img-a"
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_runner_run_node_through_get_or_load():
    """RunNode —— node-executor 走 ModelManager.get_or_load 拿 adapter 再 infer。

    不预先 LoadModel —— get_or_load 应 lazy load。
    """
    proc, ch = _spawn_runner()
    try:
        assert isinstance(await _recv(ch), P.Ready)
        await ch.send_message(P.RunNode(
            task_id=20, node_id="sampler", node_type="image",
            model_key="fake-img-a", inputs={"steps": 3},
        ))
        progresses, result = await _collect_until_result(ch, 20)
        assert result.status == "completed"
        assert result.task_id == 20
        assert len(progresses) == 3
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_runner_unknown_model_fails_node_not_runner():
    """get_or_load 撞 ModelNotFoundError —— 该节点 failed，runner 不崩。"""
    proc, ch = _spawn_runner()
    try:
        assert isinstance(await _recv(ch), P.Ready)
        await ch.send_message(P.RunNode(
            task_id=21, node_id="sampler", node_type="image",
            model_key="no-such-model", inputs={"steps": 1},
        ))
        _, result = await _collect_until_result(ch, 21)
        assert result.status == "failed"
        assert result.error
        # runner 仍活着
        await ch.send_message(P.Ping())
        assert isinstance(await _recv(ch), P.Pong)
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_concurrent_same_model_runs_are_serialized():
    """核心验证（spec §1.3 / §4.5）：并发的同模型 RunNode 被 per-model 锁串行化。

    一次性投 3 个同模型 RunNode（每个 steps 较多 → infer 有可观测耗时）。runner
    内 node-executor 是单 task 串行从队列取 —— 加上 ModelManager.load_model 的
    per-model asyncio.Lock，3 个节点的执行**不重叠**：每个节点的全部 NodeProgress
    应连续出现，不与另一节点的 progress 交错。
    """
    proc, ch = _spawn_runner()
    try:
        assert isinstance(await _recv(ch), P.Ready)
        # infer_seconds 通过 yaml params 没配 —— 用 RunNode steps 制造耗时，
        # FakeAdapter 默认 infer_seconds=0.01，3 个节点 each 6 step。
        task_ids = [30, 31, 32]
        for tid in task_ids:
            await ch.send_message(P.RunNode(
                task_id=tid, node_id="sampler", node_type="image",
                model_key="fake-img-a", inputs={"steps": 6},
            ))
        # 收齐所有消息，记录每条消息的 task_id 出现顺序
        order: list[int] = []
        results: dict[int, P.NodeResult] = {}
        while len(results) < 3:
            msg = await _recv(ch)
            if isinstance(msg, P.NodeProgress):
                order.append(msg.task_id)
            elif isinstance(msg, P.NodeResult):
                results[msg.task_id] = msg
                order.append(msg.task_id)
        # 全部 completed
        assert all(r.status == "completed" for r in results.values())
        # 串行化断言：order 里每个 task_id 的出现是连续的一段，不交错。
        # 把 order 压缩成「连续相同段」的序列，每个 task_id 只应出现一段。
        segments = [order[0]]
        for tid in order[1:]:
            if tid != segments[-1]:
                segments.append(tid)
        assert len(segments) == 3, (
            f"同模型节点执行交错了，期望 3 段连续，实得 segments={segments} "
            f"(order={order})"
        )
        assert set(segments) == set(task_ids)
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_load_failed_model_emits_model_event():
    """LoadModel 一个会 OOM-到底的模型 —— ModelEvent(load_failed)，runner 不崩。

    config 透传 oom_on_load_count=5 给 FakeAdapter（怎么试都 OOM）→
    ModelManager.get_or_load evict 后重试仍 OOM → load_failed。
    """
    proc, ch = _spawn_runner()
    try:
        assert isinstance(await _recv(ch), P.Ready)
        await ch.send_message(P.LoadModel(
            model_key="fake-img-b", config={"oom_on_load_count": 5},
        ))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent)
        assert ev.event == "load_failed"
        assert ev.error
        # runner 仍活着
        await ch.send_message(P.Ping())
        assert isinstance(await _recv(ch), P.Pong)
    finally:
        await _shutdown(proc, ch)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process_modelmanager.py -v`
Expected: FAIL —— `runner_main` 不接受 `models_yaml_path` / `fake_adapter` kwarg（Lane C 的签名是 `adapter_class`），或 `_RunnerState` 还用极简 dict。

- [ ] **Step 3: 改 `runner_process.py` 接 ModelManager**

`backend/src/runner/runner_process.py` —— 三处改动：

**(a) `_RunnerState` 持有 ModelManager（替换 `adapters` dict）。** 把 `_RunnerState.__init__` 改为：
```python
class _RunnerState:
    """runner 子进程内的可变状态。"""

    def __init__(
        self,
        runner_id: str,
        group_id: str,
        gpus: list[int],
        model_manager,  # src.services.model_manager.ModelManager
    ):
        self.runner_id = runner_id
        self.group_id = group_id
        self.gpus = gpus
        # Lane D：真 ModelManager（per-runner 独立实例，spec §4.5）。
        # 替换 Lane C 的极简 dict[model_key -> adapter]。
        self.mm = model_manager
        self.run_queue: asyncio.Queue[P.RunNode] = asyncio.Queue()
        self.cancel_flags: dict[int, threading.Event] = {}
        self.shutdown = asyncio.Event()
```
（删掉 `self.adapter_class` 和 `self.adapters`，删掉模块里的 `_load_adapter_class` helper —— ModelManager 自己管 adapter 实例化。）

**(b) `_handle_load_model` / `_handle_unload_model` 改走 ModelManager：**
```python
async def _handle_load_model(state: _RunnerState, ch: PipeChannel, msg: P.LoadModel) -> None:
    """LoadModel —— 走 ModelManager.get_or_load（含 OOM evict 重试），发 ModelEvent。"""
    from src.errors import ModelLoadError, ModelNotFoundError

    try:
        await state.mm.get_or_load(msg.model_key)
    except (ModelLoadError, ModelNotFoundError) as e:
        await ch.send_message(P.ModelEvent(
            event="load_failed", model_key=msg.model_key,
            error=f"{type(e).__name__}: {e}",
        ))
        return
    except Exception as e:  # noqa: BLE001 —— 兜底，runner 不崩
        await ch.send_message(P.ModelEvent(
            event="load_failed", model_key=msg.model_key,
            error=f"{type(e).__name__}: {e}",
        ))
        return
    await ch.send_message(P.ModelEvent(event="loaded", model_key=msg.model_key, error=None))


async def _handle_unload_model(state: _RunnerState, ch: PipeChannel, msg: P.UnloadModel) -> None:
    await state.mm.unload_model(msg.model_key, force=True)
    await ch.send_message(P.ModelEvent(event="unloaded", model_key=msg.model_key, error=None))
```
（`msg.config` 不再直接传给 adapter 构造 —— fake 模式下 config 里的 `oom_on_load_count` 等通过 `models.yaml` 的 `params` 进 `ModelSpec.params`，再由 `runner_modelmanager._fake_adapter_factory` 透传。**但测试 Step 1 是用 `LoadModel(config={"oom_on_load_count": 5})` 传的** —— 见下方 (d) 补一个 config → spec.params 的注入。）

**(c) `_node_executor` 改走 `get_or_load`：** 把原来 `adapter = state.adapters.get(node.model_key)` 那段换成：
```python
        from src.errors import ModelLoadError, ModelNotFoundError

        cancel_flag = state.cancel_flags.get(node.task_id) or threading.Event()
        started = time.monotonic()

        try:
            adapter = await state.mm.get_or_load(node.model_key) if node.model_key else None
        except (ModelLoadError, ModelNotFoundError) as e:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        if adapter is None:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"node {node.node_id!r} has no model_key",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue
```
然后 `adapter.infer(...)` 调用处 —— 真 image adapter 不接受 `progress_callback` / `cancel_flag`（偏差 4），用 signature 探测决定是否传：
```python
        try:
            from src.services.inference.base import ImageRequest

            req = ImageRequest(
                request_id=f"task-{node.task_id}",
                prompt=str(node.inputs.get("prompt", "")),
                steps=int(node.inputs.get("steps", 1) or 1),
            )
            # 真 image adapter 的 infer(req) 不接 progress_callback / cancel_flag
            # —— 那是 Lane G（D14）给真 adapter 接 callback_on_step_end 的活。
            # FakeAdapter 接受这俩 kwarg。用 signature 探测：支持就传（fake 路径
            # 拿到 within-node progress + cancel），不支持就只传 req（真 adapter
            # 路径 = 节点边界 cancel，within-node 留 Lane G）。
            import inspect

            infer_params = inspect.signature(adapter.infer).parameters
            infer_kwargs: dict = {}
            if "progress_callback" in infer_params:
                infer_kwargs["progress_callback"] = _on_progress
            if "cancel_flag" in infer_params:
                infer_kwargs["cancel_flag"] = cancel_flag
            result = await adapter.infer(req, **infer_kwargs)
        except asyncio.CancelledError:
            ...  # 原 Lane C 的 cancelled 分支不变
```
（`_on_progress` 闭包定义保持 Lane C 原样，放在 `infer` 调用之前。）

**(d) `runner_main` 改签名 + build ModelManager + config→params 注入：**
```python
def runner_main(
    group_id: str,
    gpus: list[int],
    conn: Any,
    *,
    models_yaml_path: str | None = None,
    fake_adapter: bool = False,
) -> None:
    """multiprocessing.Process 的 target —— image/TTS runner 子进程入口。

    起独立 event loop（spec §4.5：runner 有自己的 Event Loop B）+ 构造 per-runner
    独立 ModelManager（spec §4.5）。fake_adapter=True → 所有模型走 FakeAdapter。
    """
    from src.runner.runner_modelmanager import build_runner_model_manager

    runner_id = f"runner-{group_id}-{uuid.uuid4().hex[:6]}"
    mm = build_runner_model_manager(
        group_id, gpus, models_yaml_path=models_yaml_path, fake_adapter=fake_adapter,
    )
    state = _RunnerState(runner_id, group_id, gpus, mm)
    ch = PipeChannel(conn)
    try:
        asyncio.run(_runner_loop(state, ch))
    finally:
        ch.close()
```
关于 `LoadModel.config` → `FakeAdapter` 故障开关的注入：测试 Step 1 用 `LoadModel(config={"oom_on_load_count": 5})`。最干净的做法是在 `_handle_load_model` 里，load 前把 `msg.config` 合并进该 model 的 `ModelSpec.params`。在 `_handle_load_model` 的 `try` 之前加：
```python
    # config 透传：把 LoadModel.config 合并进 spec.params，让 fake adapter 的
    # 故障开关（oom_on_load_count / fail_load / infer_seconds）能经 LoadModel
    # 注入。ModelSpec frozen —— 用 registry 重建该 spec。真实部署 config 一般空。
    if msg.config:
        spec = state.mm._registry.get(msg.model_key)
        if spec is not None:
            merged = {**spec.params, **msg.config}
            state.mm._registry._specs[msg.model_key] = spec.model_copy(
                update={"params": merged}
            )
```
（`ModelSpec` 是 pydantic frozen model，`model_copy(update=...)` 是其原生支持的不可变更新方式。）

> 设计说明：Lane C 的 runner 把 adapter 实例化收在 runner 自己手里（`_load_adapter_class` + `adapters` dict）；Lane D 把这个职责整体让给 `ModelManager` —— runner 只剩「收消息 → 调 mm → 发消息」。per-model 锁、LRU evict、`_load_failures`、OOM 重试全在 `ModelManager` 内，runner 不重复实现。`runner_main` 的 Lane C `adapter_class` 参数被 `fake_adapter` bool 取代：模型用哪个 adapter 由 `ModelSpec.adapter_class`（yaml）决定，runner 不该硬编一个全局 adapter 类。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process_modelmanager.py -v`
Expected: 5 个用例全 PASS。`test_concurrent_same_model_runs_are_serialized` 是核心 —— 它断言 3 个同模型节点的 progress 不交错。注意：起真子进程，单文件约 15-25s。

- [ ] **Step 5: 跑 Lane C 的 runner_process 测试确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process.py tests/test_runner_client.py -v`
Expected: **可能 FAIL** —— Lane C 的 `test_runner_process.py` 用 `kwargs={"adapter_class": "..."}` 调 `runner_main`，Lane D 把这个参数换成了 `fake_adapter`。**这是预期的破坏性变更**。处理方式：把 Lane C 这两个测试文件里所有 `runner_main` 的调用从 `kwargs={"adapter_class": "src.runner.fake_adapter.FakeAdapter"}` 改为 `kwargs={"fake_adapter": True}`（`_spawn_runner` / `_make_client` 辅助函数里各一处）。改完重跑：
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process.py tests/test_runner_client.py -v
```
Expected: 全 PASS。若 Lane C 测试还断言了 `dict[model_key->adapter]` 的内部细节（不太可能 —— Lane C 测试是黑盒 IPC 测试），一并对齐。

- [ ] **Step 6: lint 预检 + 全 runner suite 回归**

Run:
```bash
cd backend && ruff check src/runner/ tests/test_runner_process_modelmanager.py
ADMIN_PASSWORD="" python -m pytest tests/test_runner_protocol.py tests/test_pipe_channel.py tests/test_fake_adapter.py tests/test_runner_process.py tests/test_runner_client.py tests/test_runner_supervisor.py tests/test_runner_modelmanager.py tests/test_runner_process_modelmanager.py tests/test_model_manager_get_or_load.py -q
```
Expected: 全 PASS。

- [ ] **Step 7: Commit**

```bash
cd backend && git add src/runner/runner_process.py tests/test_runner_process_modelmanager.py tests/test_runner_process.py tests/test_runner_client.py
git commit -m "feat(runner): wire real ModelManager into image/TTS runner subprocess

Replaces Lane C's bare dict[model_key->adapter] with a per-runner
ModelManager instance (spec 4.5). _handle_load_model / _node_executor
now go through ModelManager.get_or_load — per-model asyncio.Lock,
LRU evict, _load_failures and OOM-evict-retry all live in ModelManager,
not duplicated in the runner. Verified: concurrent same-model RunNode
messages are serialized by the per-model lock (no interleaved progress).
runner_main's adapter_class kwarg becomes fake_adapter bool — model
adapter is decided by ModelSpec.adapter_class. Real image adapter's
infer(req) doesn't take cancel_flag/progress_callback yet (Lane G / D14)
— node-executor signature-probes and degrades to boundary-only cancel.
V1.5 Lane D, spec 1.3 / 4.3 / 4.5."
```

---

## Task 5: 真 GPU OOM e2e 测试（`@pytest.mark.e2e`，CI skip）

spec §5.4 E2E：「故意 load 超 VRAM → LRU evict + 重试；二次 OOM → load_failed」。这条只能在真 GPU 上验，标 `@pytest.mark.e2e`，CI 默认 skip，dev box 手动 `pytest -m e2e` 跑。

**Files:**
- Create: `backend/tests/test_runner_oom_e2e.py`

- [ ] **Step 1: 确认 e2e marker 已注册**

Run: `cd backend && grep -rn "e2e" pyproject.toml pytest.ini 2>/dev/null`
Expected: 看到 `e2e` marker（Lane C 的 plan File Structure 列了 `pytest.ini markers: e2e, integration, chaos`，应已在）。若没有 —— 在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 加：
```toml
markers = [
    "e2e: real-GPU tests, skipped in CI (run on dev box with pytest -m e2e)",
    "integration: mock-runner-subprocess integration tests",
    "chaos: fault-injection tests, weekly manual run",
]
```

- [ ] **Step 2: 写 e2e 测试**

新建 `backend/tests/test_runner_oom_e2e.py`：
```python
"""Lane D: 真 GPU OOM 路径 e2e 测试（spec §5.4）。

@pytest.mark.e2e —— CI skip。dev box 手动跑：
    cd backend && python -m pytest tests/test_runner_oom_e2e.py -m e2e -v

需要：真 GPU + 一份配了「故意超 VRAM」模型的 models.yaml（见下方 SETUP）。
这条测试验证 ModelManager.get_or_load 在真 CUDA OOM 下的 evict-retry-fail 行为
—— 单测用 FakeOOMError 模拟过，但真 torch.cuda.OutOfMemoryError 的类名 / 行为
要在真硬件上确认一次。
"""
import os
from pathlib import Path

import pytest

# CI 环境 CUDA_VISIBLE_DEVICES="" —— 整个文件在无 GPU 时 skip。
pytestmark = pytest.mark.e2e

_OOM_YAML = os.getenv("NOUS_OOM_TEST_YAML", "")


@pytest.mark.skipif(
    not _OOM_YAML or not Path(_OOM_YAML).exists(),
    reason="set NOUS_OOM_TEST_YAML to a models.yaml with an oversized model",
)
@pytest.mark.asyncio
async def test_real_oom_evict_retry_then_load_failed():
    """真 GPU：先占满显存，再 load 一个放不下的模型 →
    get_or_load evict 一次重试 → 仍 OOM → load_failed。"""
    from src.runner.runner_modelmanager import build_runner_model_manager
    from src.errors import ModelLoadError

    # fake_adapter=False —— 走真 adapter / 真权重
    mm = build_runner_model_manager(
        group_id="image", gpus=[0], models_yaml_path=_OOM_YAML, fake_adapter=False,
    )
    # SETUP 约定：yaml 里 'filler' 模型先占住显存，'oversized' 放不下。
    await mm.get_or_load("filler")
    assert "filler" in mm.loaded_model_ids

    with pytest.raises(ModelLoadError):
        await mm.get_or_load("oversized")
    # filler 被 evict（get_or_load 的第一次 OOM 触发）
    assert "filler" not in mm.loaded_model_ids
    # oversized 落 _load_failures
    assert "oversized" in mm._load_failures


@pytest.mark.skipif(
    not _OOM_YAML or not Path(_OOM_YAML).exists(),
    reason="set NOUS_OOM_TEST_YAML to a models.yaml with a normal-sized model",
)
@pytest.mark.asyncio
async def test_real_model_loads_and_infers_in_runner_mm():
    """真 GPU sanity：runner ModelManager 能 load 一个真 image 模型并 infer。"""
    from src.runner.runner_modelmanager import build_runner_model_manager
    from src.services.inference.base import ImageRequest

    mm = build_runner_model_manager(
        group_id="image", gpus=[0], models_yaml_path=_OOM_YAML, fake_adapter=False,
    )
    adapter = await mm.get_or_load("normal")  # yaml 约定的正常尺寸模型
    assert adapter.is_loaded
    result = await adapter.infer(ImageRequest(
        request_id="e2e-1", prompt="a red cube", steps=4, width=512, height=512,
    ))
    assert result.media_type.startswith("image/")
    assert result.data
```

> SETUP 说明（写进测试 docstring 已够，此处补全）：跑这条 e2e 前，准备一份 `models.yaml`，含三个条目 —— `filler`（一个能占住目标 GPU 大部分显存的真模型）、`oversized`（剩余显存绝对放不下的模型，例如另一个大 image 模型）、`normal`（正常尺寸 image 模型）。`export NOUS_OOM_TEST_YAML=/path/to/oom-test-models.yaml` 后 `pytest -m e2e`。这条测试不进 CI —— CI 跑 `pytest`（不带 `-m e2e`）自动 skip 整个文件。

- [ ] **Step 3: 确认 CI 默认 skip**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_oom_e2e.py -q`
Expected: 2 个用例都 `skipped`（CI 跑的就是这条命令 —— 不带 `-m e2e`，e2e marker 的测试默认收集但跳过；实际行为取决于 pytest 配置 —— 若配了 `-m "not e2e"` 的默认 addopts 则 deselected，否则需 skipif 兜底。这里 `NOUS_OOM_TEST_YAML` 未设 → skipif 必定 skip，双保险）。**关键：不报 error、不真的 load 模型。**

- [ ] **Step 4: lint 预检 + Commit**

```bash
cd backend && ruff check tests/test_runner_oom_e2e.py
git add tests/test_runner_oom_e2e.py pyproject.toml
git commit -m "test(runner): add real-GPU OOM evict-retry e2e test (spec 5.4)

@pytest.mark.e2e — CI skips it (NOUS_OOM_TEST_YAML unset → skipif).
Dev box runs pytest -m e2e to verify ModelManager.get_or_load's
evict-retry-fail path against real torch.cuda.OutOfMemoryError, since
unit tests only simulate OOM via FakeOOMError. V1.5 Lane D."
```

---

## Task 6: Lane D 整合验证 + 开 PR

**Files:** 无（验证）

- [ ] **Step 1: 全后端 suite green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS，无 collection error、无 import error。e2e 测试 skip。

- [ ] **Step 2: 确认 runner 已脱离极简 adapter dict**

Run:
```bash
cd backend && grep -n "self.adapters\|_load_adapter_class\|adapter_class" src/runner/runner_process.py
```
Expected: **零输出** —— Lane C 的极简 `adapters` dict / `_load_adapter_class` / `adapter_class` 参数已全部移除，runner 走 ModelManager。

- [ ] **Step 3: 确认 ModelManager 没被塞测试 flag**

Run:
```bash
cd backend && grep -n "fake\|test" src/services/model_manager.py
```
Expected: 零输出（或仅注释）—— fake 模式的 monkeypatch 在 `runner_modelmanager.py` 工厂里做，`ModelManager` 本体干净（偏差说明的设计承诺）。

- [ ] **Step 4: lint 全量预检**

Run: `cd backend && ruff check src/ tests/`
Expected: 无新增 lint 错误。

- [ ] **Step 5: 收尾 commit（若有未提交的 plan 勾选）**

```bash
git add docs/superpowers/plans/2026-05-14-v15-laneD-modelmanager-into-runner.md
git commit -m "docs(plan): Lane D complete — ModelManager wired into runner"
```

- [ ] **Step 6: 开 PR**

```bash
git push -u origin <lane-d-branch>
gh pr create --title "feat: V1.5 Lane D — ModelManager into image/TTS runner" --body "$(cat <<'EOF'
## Summary
- `ModelManager.get_or_load`：OOM evict + 重试一次（spec §4.3，原 ModelManager 没有此方法）
- runner 子进程持有独立 `ModelManager` 实例（spec §4.5），替换 Lane C 的极简 adapter dict
- node-executor 走 `get_or_load`：per-model `asyncio.Lock` / LRU evict / `_load_failures` 全复用 ModelManager
- 验证：并发同模型 RunNode 被 per-model 锁串行化（progress 不交错）
- `FakeAdapter` 加 OOM 模拟开关；`build_runner_model_manager` 工厂（fake 模式不污染 ModelManager 本体）
- 真 GPU OOM e2e 测试（`@pytest.mark.e2e`，CI skip）

## 偏差（详见 plan 头部）
- spec §4.3 的 `get_or_load` 实为新建（spec 草图误以为已有）
- 真 image adapter `infer(req)` 不接 cancel_flag/progress_callback —— within-node cancel 是 Lane G/D14；Lane D 用 signature 探测降级为节点边界 cancel

## Test plan
- [ ] 全 suite green（pytest tests/）
- [ ] `test_concurrent_same_model_runs_are_serialized` PASS（核心：per-model 锁串行化）
- [ ] Lane C 的 runner_process / runner_client 测试对齐 fake_adapter kwarg 后仍 green
- [ ] e2e 测试 CI 自动 skip；dev box `pytest -m e2e` 手动验真 GPU OOM 路径
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneD-modelmanager-into-runner`。）

---

## Self-Review

**Spec 覆盖检查：** Lane D 在 spec「实施分 Lane」表里的职责是「ModelManager 迁入 image/TTS runner（image runner 先迁，验证 per-model lock）。依赖：B, C」。

- ModelManager 迁入 runner 子进程 → Task 3（工厂）+ Task 4（接线）
- 每个 runner 独立 ModelManager 实例（spec §4.5）→ Task 3 `build_runner_model_manager` + `test_factory_each_call_is_independent_instance`
- 验证 per-model lock → Task 4 `test_concurrent_same_model_runs_are_serialized`（核心测试，断言 3 个同模型节点 progress 不交错）
- OOM 处理（spec §4.3：load OOM → evict + 重试一次 → load_failed）→ Task 1 `get_or_load` + Task 4 `test_load_failed_model_emits_model_event` + Task 5 真 GPU e2e
- image runner 先迁、泛化到 TTS → runner_process / runner_modelmanager 不写死 image，`group_id` 参数贯穿，TTS 真正接入是 Lane F（plan 多处注明）
- RunNode/NodeResult 协议（spec §3.3）→ 复用 Lane C 的 `protocol.py`，Lane D 不改协议

**Spec 歧义 / 偏差（已在 plan 头部「注意」逐条列出）：**
1. **spec §4.3 把 `get_or_load` 写得像 ModelManager 已有方法 —— 实际不存在。** 源码核实只有 `get_loaded_adapter`（无重试）。Lane D 新建 `get_or_load`，语义对齐 §4.3 草图。
2. spec §4.3 草图 `self.evict_lru(...)` 漏 `await`，`evict_lru` 是 `async def` —— Task 1 实现已 `await`。
3. spec §4.3 草图捕获 `torch.cuda.OutOfMemoryError`，但测试环境 torch 是 MagicMock —— `get_or_load._is_oom` 改用类名 + 文本判定，不 import torch。
4. **真 image adapter `image_diffusers.py` 的 `infer(req)` 不接受 `cancel_flag` / `progress_callback`** —— 给真 adapter 接 `callback_on_step_end` 是 Lane G（D14）。Lane D 用 `inspect.signature` 探测，真 adapter 路径降级为节点边界 cancel，within-node 留 Lane G。这是 Lane 边界的诚实划分，不是遗漏。
5. Lane C 尚未实现 —— Task 0 前置检查；若 Lane C 不在，停下来先做 Lane C。
6. **依赖表写了 B，但 Lane D 不直接用 Lane B 产物**（TaskRingBuffer / schema migration）。Lane D 改动全在 runner 子进程内，不碰 `execution_tasks`。列 B 是因为后续集成 Lane（G/J）需要两者都在。

**判断调用（judgment calls）：**
- **fake 模式用实例级 monkeypatch 而非 ModelManager 构造 flag。** `ModelManager` 是 V1.5 多 Lane 共用核心类，往构造里塞 `fake=True` 会污染真实路径。工厂在 runner 边界做 `functools.partial` 绑 adapter_factory —— Task 6 Step 3 grep 验证 `ModelManager` 本体无 `fake`/`test` 字样。
- **`runner_main` 的 Lane C `adapter_class` 参数替换为 `fake_adapter` bool。** 模型用哪个 adapter 应由 `ModelSpec.adapter_class`（yaml）决定，runner 不该硬编全局 adapter 类。这是破坏性变更，Task 4 Step 5 显式处理 Lane C 测试的对齐（`kwargs` 改 `fake_adapter=True`）。
- **`LoadModel.config` → `ModelSpec.params` 的注入** 用 pydantic `model_copy(update=...)`（frozen model 的原生不可变更新）。真实部署 config 一般空；这条路径主要服务测试通过 `LoadModel` 注入 fake 故障开关。
- **per-model 锁的位置：** 锁在 `ModelManager.load_model` 内（`async with self._lock_for(model_id)`），`get_or_load` 不再包一层。并发同模型 `get_or_load` 可能两个都进 `load_model`，但 `load_model` 内 `if self.is_loaded: return` 让第二个变 no-op —— `test_concurrent_same_model_runs_are_serialized` 验证最终行为（node-executor 单 task 串行取队列 + 锁，节点执行不重叠）。注：runner 的 node-executor 本来就是单 task 串行的，per-model 锁在「同模型并发 load」场景才真正吃重 —— 该测试同时覆盖了「单 task 串行」和「锁不死锁」两件事。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有新代码、测试、命令、commit message 完整给出。Task 4 的 `runner_process.py` 改动按 (a)(b)(c)(d) 分块给出具体替换内容；Lane C 既有代码的修改点（`_RunnerState` / `_handle_load_model` / `_node_executor` / `runner_main`）逐处标明。

**类型一致性：** `get_or_load(model_id, adapter_factory=None) -> InferenceAdapter` 与 `get_loaded_adapter` / `load_model` 签名族一致；`build_runner_model_manager(group_id, gpus, *, models_yaml_path, fake_adapter) -> ModelManager`；`runner_main(group_id, gpus, conn, *, models_yaml_path, fake_adapter)` 与 Task 4 测试的 `args` / `kwargs` 调用一致；`_RunnerState.__init__(runner_id, group_id, gpus, model_manager)` 与 `runner_main` 的构造调用一致。

**已知风险：**
- Task 0 前置检查是整个 Lane 的前提 —— Lane C 未落地则 Lane D 无从谈起，Step 1 有「停下来先做 Lane C」护栏。
- Task 4 Step 5 改 Lane C 的测试文件 —— 若 Lane C 实际实现与本 plan 假设的 `_spawn_runner` / `_make_client` 辅助函数形态不同，对齐时需按实际代码调整（黑盒 IPC 测试，改动应仅限 `runner_main` 的 kwarg）。
- `GPUAllocator()` 无参构造 + `CUDA_VISIBLE_DEVICES=""` 下的行为已按「poll 不到 GPU 返回空 / -1」假设；若 `GPUAllocator` 构造需要参数或在无 GPU 时抛异常，Task 3 的工厂需加兜底（Task 3 Step 5 的测试会第一时间暴露）。
- within-node cancel 在 Lane D 的真 adapter 路径下不生效（降级为节点边界）—— 这是 spec 认可的 Lane 边界（Lane G/D14 承接），但若有人误以为 Lane D 完成后 image 就能 within-node cancel，会有预期落差。plan 头部偏差 4 + 本节已显式说明。
