# V1.5 Lane G: GroupScheduler + Cancel 双层 + Image Adapter 重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 V1.5 调度的三件相互咬合的事：(1) `GroupScheduler` —— 主进程内每 GPU group 一个 `asyncio.PriorityQueue` 派发器，2 级优先级（interactive=0 / batch=10），同级 `queued_at` FIFO；(2) Cancel 双层 —— 节点边界 cancel（`cancel_event` 在每个节点 dispatch 前 check）+ within-node cancel（image sampler 每 step）；(3) **image adapter 重写（D14 / G1 / G2）** —— 现状 `image_diffusers.py` 把 diffusers `pipe.__call__()` 当 `to_thread` 里一个不透明阻塞调用，没有 per-step 回调，`asyncio.wait_for` 取消 `to_thread` 不会停 CUDA kernel。重写后接入 diffusers `callback_on_step_end`，让一个 `threading.Event`（runner 的 pipe-reader 收到 Abort 时 set，或 `asyncio.wait_for` 超时时 set）能在采样步之间中断扩散循环、真正停掉 CUDA kernel。cancel 与 timeout 走**同一个** flag。

**Architecture:** 三块交付物，依赖链 adapter 重写 → CancelFlag → GroupScheduler：

1. **`CancelFlag`（`src/services/inference/cancel_flag.py` 新建）** —— 一个极薄的 `threading.Event` 包装。它是 cancel 信号穿过 `to_thread` 边界的唯一载体。pipe-reader（runner 内，Lane C）收到 `Abort` → `flag.set()`；`asyncio.wait_for` 超时 → `flag.set()`；callback 在 `to_thread` 工作线程里 check `flag.is_set()` → 抛 `NodeCancelled`。放 `inference/` 下而非 `scheduler/` 下，因为 adapter 和 runner 都要 import 它，scheduler 反而不直接碰。

2. **Image adapter 重写（`image_diffusers.py` 改 `sample()` + `infer()`）** —— `sample()` 与 `infer()` 接受可选 `cancel_flag` 参数，构造 diffusers `callback_on_step_end` 闭包：每步 check `cancel_flag.is_set()`，命中则 `raise NodeCancelled`。`infer()` 把 `pipe.__call__` 包进 `asyncio.wait_for(asyncio.to_thread(...), timeout=...)`，超时分支 `cancel_flag.set()` 后 `raise NodeTimeout` —— 让已在飞的 `to_thread` 工作线程在下一采样步自行中断（不能直接杀线程，CUDA kernel 不受 Python 线程取消影响）。`NodeCancelled` / `NodeTimeout` 是本 Lane 新建的两个 exception（`inference/exceptions.py`）。TTS adapter **不改**（spec §4.4：TTS / VAE / 短节点只在边界 check，within-node check 的 overhead 比收益大）。

3. **`GroupScheduler`（`src/services/scheduler/group_scheduler.py` 新建）** —— spec §3.5 的 `QueuedTask` + `GroupScheduler` dataclass。`QueuedTask` 是 `@dataclass(order=True)`，`sort_key=(priority, queued_at)` 参与比较，`task_id` / `workflow_spec` 标 `compare=False`。`GroupScheduler` 持有一个 `asyncio.PriorityQueue[QueuedTask]`、一个 dispatcher loop（弹最小者 → 标 running → 调 executor → 终态回收）、`cancel_events: dict[int, asyncio.Event]`、`inflight_tasks: dict[int, asyncio.Task]`。队列堆积 >1000 时 `enqueue` 抛 `QueueFullError`（路由层转 503 + Retry-After，路由接线属 Lane S，本 Lane 只抛异常 + 单测）。

**Lane G 不做（划清边界）：**
- **不**写 runner 子进程 / pipe-reader / RunnerClient —— 那是 Lane C。本 Lane 的 `GroupScheduler` 通过一个**注入的 executor 回调**（`Callable[[int, dict, asyncio.Event, CancelFlag], Awaitable[dict]]`）跑任务，单测里用 fake executor；真 executor 接线（dispatch 节点 → RunnerClient）是 Lane S。
- **不**改 `/v1/workflows/{id}/run` 路由契约、**不**改 `/v1/tasks/{id}/cancel` 路由 —— 那是 Lane S。本 Lane 只提供 `GroupScheduler.request_cancel(task_id, reason)` API + 单测。
- **不**接 DB / `TaskRingBuffer` —— 那是 Lane S/I 把 scheduler 接进主进程时的事。本 Lane 的 `GroupScheduler` 是纯内存逻辑，`queued_at` 由调用方传入。
- **不**碰 LLM —— LLM group 无 dispatch 队列（spec §1.2），`GroupScheduler.runner_client` 对 LLM group 为 `None`；本 Lane 只测 image/TTS 形态的 scheduler。

**Tech Stack:** Python 3.12 / asyncio（`PriorityQueue` / `Event` / `wait_for` / `to_thread`）/ `threading.Event` / `dataclasses`（`order=True` / `field(compare=False)`）/ pytest（`asyncio_mode = "auto"`）。adapter 测试用 fake diffusers pipe（纯 Python，invoke callback，**不需要真 GPU**）—— `conftest.py` 已强制 `CUDA_VISIBLE_DEVICES=""`，不会误碰 CUDA。

> **注意 — 与 spec 的偏差 / 模糊处（已核实，须知会）：**
>
> 1. **`QueuedTask.sort_key` 用 `datetime` 直接参与比较有歧义。** spec §3.5 写 `sort_key: tuple[int, datetime]`，但 §2.2 又写「入队 sort key = `(priority, queued_at, task_id)`」—— 三元组。**判断：用 `(priority, queued_at, task_id)` 三元组**。理由：纯 `(priority, datetime)` 在两个 task 同优先级且 `queued_at` 精度内相等时，`PriorityQueue` 会尝试比较 `QueuedTask` 本身（`dataclass(order=True)` 下会继续比 `task_id` 之后的 compare 字段，而 `workflow_spec` dict 不可比 → `TypeError`）。加 `task_id` 作第三 key 保证 sort_key 元组永远可全序比较，`QueuedTask` 的其余字段全部 `compare=False`。已在 §3.5 实现 + Self-Review flag。
>
> 2. **`asyncio.PriorityQueue` 不暴露「按 task_id 删除」。** spec §4.7 case 1「cancel 已 dispatch 但 runner 还没收到」靠 `cancel_event` + 丢弃 NodeResult 解决（不删队列）；但「cancel 还在排队、未 dispatch 的 task」spec 没明说怎么从 `PriorityQueue` 里摘掉。**判断：不从队列物理删除**，改为 dispatcher 弹出时 check `cancel_event.is_set()`，命中则直接标 `cancelled` 跳过、不调 executor（与 `task_queue.py` 旧实现的 `_cancelled` 标志同思路）。已在 §3.5 dispatcher 实现 + Self-Review flag。
>
> 3. **Lane C 尚未落地，`RunnerClient` 接口未定。** spec「实施分 Lane」表写 Lane G「依赖：C」，但截至本 plan 编写，Lane C plan 未写、`RunnerClient` 不存在。**判断：本 Lane 用注入的 executor 回调解耦** —— `GroupScheduler` 不直接 import `RunnerClient`，构造时接收一个 `executor` async callable。真 executor（内部调 `RunnerClient.run_node`）由 Lane S 注入。这样 Lane G 可独立实现 + 独立测试，不阻塞在 Lane C 上。已在 Architecture「Lane G 不做」+ Self-Review flag。
>
> 4. **spec §4.4 的 `_on_step_end` 闭包捕获 `cancel_flag` 是模块级伪代码**，真实现里 `cancel_flag` 必须按调用绑定（每次 `infer()` 一个新 flag），不能模块级共享。已在 §3.3 实现里用闭包工厂 `_make_step_callback(cancel_flag)` 落实。

---

## File Structure

| 文件 | Lane G 动作 | 责任 |
|---|---|---|
| `backend/src/services/inference/exceptions.py` | **新建** | `NodeCancelled` / `NodeTimeout` / `QueueFullError` 三个 exception |
| `backend/src/services/inference/cancel_flag.py` | **新建** | `CancelFlag` —— `threading.Event` 薄包装，cancel 信号穿 `to_thread` 的唯一载体 |
| `backend/src/services/inference/image_diffusers.py` | **修改** | `sample()` 加 `cancel_flag` 参数 + `callback_on_step_end` 闭包；`infer()` 包 `wait_for(to_thread(...))`，cancel/timeout 共用 flag |
| `backend/src/services/scheduler/__init__.py` | **新建** | 空包标记 |
| `backend/src/services/scheduler/group_scheduler.py` | **新建** | `QueuedTask` + `GroupScheduler`（PriorityQueue 派发 + cancel_events + inflight + 堆积 503） |
| `backend/tests/test_cancel_flag.py` | **新建** | `CancelFlag` set/is_set/clear 行为 |
| `backend/tests/test_image_adapter_cancel.py` | **新建** | fake diffusers pipe + callback 中断（cancel 路径 + timeout 路径 + 正常完成） |
| `backend/tests/test_group_scheduler.py` | **新建** | 优先级排序 / 同级 FIFO / cancel 排队中 / cancel inflight / 堆积 503 / inflight 回收 |
| `backend/pyproject.toml` | **修改** | `[tool.pytest.ini_options]` 加 `markers`（`integration` / `e2e` / `chaos`，spec §5.6 要求；本 Lane 全是 unit 不打 marker，但先把 markers 注册掉避免后续 Lane 的 `-m` 报 unknown marker） |

---

## Task 1: `NodeCancelled` / `NodeTimeout` / `QueueFullError` exception

三个 exception 是后面所有 Task 的共用词汇：adapter 的 callback 抛 `NodeCancelled`，`infer()` 超时抛 `NodeTimeout`，`GroupScheduler.enqueue` 堆积抛 `QueueFullError`。先把它们立起来，零逻辑。

**Files:**
- Create: `backend/src/services/inference/exceptions.py`
- Test: 无独立测试（exception 仅是标记类，由 Task 3 / Task 5 的测试间接覆盖）

- [ ] **Step 1: 创建 exception 模块**

新建 `backend/src/services/inference/exceptions.py`：
```python
"""V1.5 Lane G —— 调度 / cancel 路径的 exception 词汇。

三个 exception 跨 Lane G 三块交付物共用：
  * NodeCancelled  —— within-node cancel：diffusers callback_on_step_end 检测到
                      CancelFlag 已 set，中断扩散循环时抛。也用于节点边界 cancel。
  * NodeTimeout    —— 节点执行超时：asyncio.wait_for 包住的 to_thread 超时，
                      set CancelFlag 后抛（让在飞的工作线程下一步自行中断）。
  * QueueFullError —— GroupScheduler 队列堆积超过阈值（spec §4.7 case 4），
                      enqueue 拒绝新 task；路由层（Lane S）转 503 + Retry-After。
"""
from __future__ import annotations


class NodeCancelled(Exception):
    """节点被取消（边界 cancel 或 within-node cancel）。

    携带 reason 便于落 ExecutionTask.cancel_reason（spec §3.1）。
    """

    def __init__(self, reason: str = "cancelled"):
        self.reason = reason
        super().__init__(reason)


class NodeTimeout(Exception):
    """节点执行超时。timeout_s 是触发超时的阈值，便于日志 / cancel_reason。"""

    def __init__(self, timeout_s: float, reason: str = "node timeout"):
        self.timeout_s = timeout_s
        self.reason = reason
        super().__init__(f"{reason} after {timeout_s}s")


class QueueFullError(Exception):
    """某 GPU group 的 PriorityQueue 堆积超过 capacity，拒绝新入队。

    retry_after_s 给路由层 (Lane S) 拼 Retry-After 响应头用。
    """

    def __init__(self, group_id: str, capacity: int, retry_after_s: int = 30):
        self.group_id = group_id
        self.capacity = capacity
        self.retry_after_s = retry_after_s
        super().__init__(
            f"group {group_id!r} queue full (capacity={capacity}), "
            f"retry after {retry_after_s}s"
        )
```

- [ ] **Step 2: import 自检**

Run: `cd backend && ADMIN_PASSWORD="" python -c "from src.services.inference.exceptions import NodeCancelled, NodeTimeout, QueueFullError; e=NodeTimeout(30.0); assert e.timeout_s==30.0; q=QueueFullError('image', 1000); assert q.retry_after_s==30; print('exceptions OK')"`
Expected: `exceptions OK`

- [ ] **Step 3: Commit**

```bash
cd backend && git add src/services/inference/exceptions.py
git commit -m "feat(inference): add NodeCancelled/NodeTimeout/QueueFullError exceptions

Shared vocabulary across Lane G's three deliverables — adapter
callback raises NodeCancelled, infer() timeout raises NodeTimeout,
GroupScheduler.enqueue raises QueueFullError on backlog. V1.5 Lane G."
```

---

## Task 2: `CancelFlag` —— cancel 信号穿 to_thread 的载体

spec §4.4 的核心机制：`asyncio.wait_for` 取消的是 awaiting，`to_thread` 里的 CUDA kernel 照跑。唯一能穿过 `to_thread` 边界让工作线程自行中断的是一个跨线程的 `threading.Event`。`CancelFlag` 就是它的薄包装 —— 给它一个名字 + 一个 `reason` 字段，便于多处（pipe-reader Abort / wait_for timeout）set 时记录是谁触发的。

**Files:**
- Create: `backend/src/services/inference/cancel_flag.py`
- Test: `backend/tests/test_cancel_flag.py`（新建）

- [ ] **Step 1: 写失败测试 —— CancelFlag 行为**

新建 `backend/tests/test_cancel_flag.py`：
```python
"""Lane G: CancelFlag —— threading.Event 薄包装，cancel 信号穿 to_thread 的载体。"""
import threading

from src.services.inference.cancel_flag import CancelFlag


def test_initial_state_not_set():
    flag = CancelFlag()
    assert flag.is_set() is False
    assert flag.reason is None


def test_set_records_reason():
    flag = CancelFlag()
    flag.set("user requested")
    assert flag.is_set() is True
    assert flag.reason == "user requested"


def test_set_default_reason():
    flag = CancelFlag()
    flag.set()
    assert flag.is_set() is True
    assert flag.reason == "cancelled"


def test_set_is_idempotent_first_reason_wins():
    """多处可能都 set（pipe-reader Abort + wait_for timeout 竞态）；
    第一个 reason 留下，后续 set 不覆盖 —— 便于事后判定真正触发源。"""
    flag = CancelFlag()
    flag.set("node timeout")
    flag.set("user requested")
    assert flag.reason == "node timeout"


def test_clear_resets():
    flag = CancelFlag()
    flag.set("x")
    flag.clear()
    assert flag.is_set() is False
    assert flag.reason is None


def test_visible_across_threads():
    """worker 线程里 set，主线程能看到 —— 这是它存在的全部理由
    （to_thread 工作线程 set / 主线程 poll，或反之）。"""
    flag = CancelFlag()
    seen = []

    def worker():
        flag.set("from worker")
        seen.append(flag.is_set())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [True]
    assert flag.is_set() is True
    assert flag.reason == "from worker"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_cancel_flag.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.inference.cancel_flag'`

- [ ] **Step 3: 实现 `CancelFlag`**

新建 `backend/src/services/inference/cancel_flag.py`：
```python
"""CancelFlag —— cancel / timeout 信号穿过 asyncio.to_thread 边界的唯一载体。

spec §4.4 的关键性质：asyncio.wait_for 取消的是 awaiting，to_thread 工作线程里
的 CUDA kernel 照跑。要真正停掉 kernel，必须让工作线程**自己**在两个 kernel
launch 之间检查一个跨线程可见的标志并主动 raise。CancelFlag 就是这个标志。

谁会 set 它：
  * runner 的 pipe-reader 收到 Abort 消息（Lane C）
  * adapter.infer() 里 asyncio.wait_for 超时（Lane G，本 Lane）
两条路径 set 的是同一个 flag —— cancel 与 timeout 共用一套中断机制。

谁会读它：
  * image adapter 的 diffusers callback_on_step_end（每采样步 check 一次）

为什么不直接用裸 threading.Event：需要一个 reason 字段，事后能判定到底是
「用户取消」还是「超时」—— 直接决定 ExecutionTask.cancel_reason 落什么。
"""
from __future__ import annotations

import threading


class CancelFlag:
    """跨线程 cancel 标志 + reason。线程安全（threading.Event 本身线程安全，
    reason 的首次写入用同一把锁保护，保证「first reason wins」）。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def set(self, reason: str = "cancelled") -> None:
        """置位。多处竞态 set 时，第一个 reason 留下（便于判定真正触发源）。"""
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        """复位（adapter 复用场景；正常一次 infer 一个新 flag，少用）。"""
        with self._lock:
            self._event.clear()
            self._reason = None

    @property
    def reason(self) -> str | None:
        return self._reason
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_cancel_flag.py -v`
Expected: 6 个用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/inference/cancel_flag.py tests/test_cancel_flag.py
git commit -m "feat(inference): add CancelFlag — cancel signal carrier across to_thread

threading.Event wrapper with a reason field. The ONLY mechanism that
can carry a cancel/timeout signal across the asyncio.to_thread boundary
and let the worker thread self-interrupt between CUDA kernel launches
(spec 4.4). pipe-reader Abort and wait_for timeout both set the same
flag. V1.5 Lane G."
```

---

## Task 3: image adapter 重写 —— 接入 `callback_on_step_end`

这是 Lane G 风险最高的一块（spec GSTACK REVIEW 标为 G2 CRITICAL GAP）。现状（见 `image_diffusers.py` 当前 `sample()` :180 和 `infer()` :561）：`pipe.__call__` 是 `to_thread` 里一个不透明阻塞调用，没有 per-step 回调。重写后：

- `sample()` 加可选 `cancel_flag: CancelFlag | None` 参数，构造一个 `callback_on_step_end` 闭包，每步 check `cancel_flag.is_set()`，命中 `raise NodeCancelled`。diffusers 0.38 的 `callback_on_step_end` 签名是 `(pipe, step, timestep, callback_kwargs) -> dict`。
- `infer()` 构造一个 per-call `CancelFlag`，把 `pipe.__call__` 包进 `asyncio.wait_for(asyncio.to_thread(_run), timeout=...)`；超时分支 `cancel_flag.set("node timeout")` 后 `raise NodeTimeout` —— **不**杀线程（杀不掉 CUDA kernel），靠 callback 在下一步自行中断。
- `infer()` 接受一个可选外部 `cancel_flag` 参数（runner 的 pipe-reader 持有同一个引用，收到 Abort 时 set 它）。外部没传则内部自建（兼容 V1 直调路径）。

> 注意：本 Lane **只重写 image adapter**。TTS adapter（`tts_engines/base.py`）不动 —— spec §4.4 明确 TTS / VAE / 短节点只在边界 check，within-node 的 per-step check overhead 比收益大。

**Files:**
- Modify: `backend/src/services/inference/image_diffusers.py`（`sample()` + `infer()`，新增一个 `_make_step_callback` 模块级 helper）
- Test: `backend/tests/test_image_adapter_cancel.py`（新建，用 fake diffusers pipe，不碰 GPU）

- [ ] **Step 1: 写失败测试 —— fake pipe + callback 三条路径**

新建 `backend/tests/test_image_adapter_cancel.py`：
```python
"""Lane G: image adapter callback_on_step_end 接入测试。

用 fake diffusers pipe（纯 Python，模拟 num_inference_steps 步、每步 invoke
callback_on_step_end），不需要真 GPU。覆盖三条路径：
  1. 正常完成 —— callback 每步返回，跑满 steps
  2. within-node cancel —— 第 N 步 CancelFlag 被 set，callback 抛 NodeCancelled
  3. timeout —— wait_for 超时，set flag，下一步 callback 抛，infer 抛 NodeTimeout
"""
import asyncio

import pytest

from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.exceptions import NodeCancelled, NodeTimeout
from src.services.inference.image_diffusers import _make_step_callback


# ---- _make_step_callback 单元测试（纯函数，最快的反馈） -------------------

def test_step_callback_passthrough_when_not_cancelled():
    """flag 未 set 时，callback 原样返回 callback_kwargs，不抛。"""
    flag = CancelFlag()
    cb = _make_step_callback(flag)
    kwargs = {"latents": "fake-tensor"}
    out = cb(None, step=3, timestep=100, callback_kwargs=kwargs)
    assert out is kwargs  # 原样透传


def test_step_callback_raises_when_cancelled():
    """flag 已 set 时，callback 抛 NodeCancelled，reason 来自 flag。"""
    flag = CancelFlag()
    flag.set("user requested")
    cb = _make_step_callback(flag)
    with pytest.raises(NodeCancelled) as ei:
        cb(None, step=5, timestep=80, callback_kwargs={})
    assert ei.value.reason == "user requested"


def test_step_callback_none_flag_never_raises():
    """cancel_flag=None（V1 兼容路径）时 callback 永不抛。"""
    cb = _make_step_callback(None)
    out = cb(None, step=1, timestep=10, callback_kwargs={"x": 1})
    assert out == {"x": 1}


# ---- fake pipe 集成测试 ---------------------------------------------------

class _FakePipe:
    """模拟 diffusers pipe.__call__ 的采样循环：跑 num_inference_steps 步，
    每步 invoke callback_on_step_end。完全同步、纯 Python、不碰 CUDA。"""

    def __init__(self, step_sleep_s: float = 0.0):
        self.vae_scale_factor = 8
        self._step_sleep_s = step_sleep_s
        self.steps_run = 0

    def __call__(self, *, num_inference_steps, callback_on_step_end=None, **kw):
        import time
        for step in range(num_inference_steps):
            if callback_on_step_end is not None:
                # diffusers 真实签名: (pipe, step, timestep, callback_kwargs)
                callback_on_step_end(self, step, 1000 - step, {})
            self.steps_run += 1
            if self._step_sleep_s:
                time.sleep(self._step_sleep_s)
        return ([f"latent-after-{num_inference_steps}-steps"],)


@pytest.mark.asyncio
async def test_sample_runs_all_steps_when_not_cancelled():
    """正常路径：cancel_flag 未 set，sample 跑满 num_inference_steps。"""
    from src.services.inference.image_diffusers import sample

    pipe = _FakePipe()
    flag = CancelFlag()
    conditioning = {"prompt_embeds": "fake", "text_ids": "fake"}
    # sample 是同步的（spec 注释：node executor 在单个 to_thread 块里组合）；
    # 直接调，验证 callback 被装上且不抛。
    result = await asyncio.to_thread(
        sample, pipe, conditioning,
        width=512, height=512, num_inference_steps=8,
        guidance_scale=3.5, cancel_flag=flag,
    )
    assert pipe.steps_run == 8
    assert result is not None


@pytest.mark.asyncio
async def test_sample_interrupts_when_flag_set_mid_run():
    """within-node cancel：跑到第 3 步时另一线程 set flag，sample 在下一步抛
    NodeCancelled，steps_run 远小于请求的 30。"""
    from src.services.inference.image_diffusers import sample

    pipe = _FakePipe(step_sleep_s=0.01)  # 给取消线程一个介入窗口
    flag = CancelFlag()

    async def cancel_after_delay():
        await asyncio.sleep(0.05)  # ~5 步后
        flag.set("user requested")

    conditioning = {"prompt_embeds": "fake", "text_ids": "fake"}
    cancel_task = asyncio.create_task(cancel_after_delay())
    with pytest.raises(NodeCancelled):
        await asyncio.to_thread(
            sample, pipe, conditioning,
            width=512, height=512, num_inference_steps=30,
            guidance_scale=3.5, cancel_flag=flag,
        )
    await cancel_task
    assert pipe.steps_run < 30  # 没跑满 —— 被中断了
    assert pipe.steps_run >= 1


@pytest.mark.asyncio
async def test_infer_timeout_sets_flag_and_raises_node_timeout():
    """timeout 路径：infer 把采样包进 wait_for(to_thread(...))，超时后 set
    flag + 抛 NodeTimeout；在飞的 fake pipe 在下一步因 flag 中断（不挂死）。"""
    from src.services.inference.image_diffusers import DiffusersImageBackend
    from src.services.inference.base import ImageRequest

    adapter = DiffusersImageBackend(paths={"main": "/fake"}, device="cpu")
    # 直接注入 fake pipe，跳过 load()
    adapter._pipe = _FakePipe(step_sleep_s=0.05)  # 每步 50ms，30 步 = 1.5s

    req = ImageRequest(
        request_id="t-timeout", prompt="x",
        width=512, height=512, steps=30, cfg_scale=3.5,
        timeout_s=0.2,  # 0.2s 远小于 1.5s —— 必超时
    )
    with pytest.raises(NodeTimeout) as ei:
        await adapter.infer(req)
    assert ei.value.timeout_s == 0.2


@pytest.mark.asyncio
async def test_infer_external_cancel_flag_is_honored():
    """infer 接受外部传入的 cancel_flag（runner pipe-reader 持同一引用）；
    外部 set 后，在飞的采样在下一步中断，infer 抛 NodeCancelled。"""
    from src.services.inference.image_diffusers import DiffusersImageBackend
    from src.services.inference.base import ImageRequest

    adapter = DiffusersImageBackend(paths={"main": "/fake"}, device="cpu")
    adapter._pipe = _FakePipe(step_sleep_s=0.02)

    external_flag = CancelFlag()
    req = ImageRequest(
        request_id="t-ext-cancel", prompt="x",
        width=512, height=512, steps=30, cfg_scale=3.5,
    )

    async def cancel_soon():
        await asyncio.sleep(0.1)
        external_flag.set("aborted by runner")

    cancel_task = asyncio.create_task(cancel_soon())
    with pytest.raises(NodeCancelled) as ei:
        await adapter.infer(req, cancel_flag=external_flag)
    await cancel_task
    assert ei.value.reason == "aborted by runner"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_image_adapter_cancel.py -v`
Expected: FAIL —— `ImportError: cannot import name '_make_step_callback' from 'src.services.inference.image_diffusers'`（以及 `sample` / `infer` 还没有 `cancel_flag` 参数）。

- [ ] **Step 3: 加 `_make_step_callback` 模块级 helper**

在 `backend/src/services/inference/image_diffusers.py` 顶部 import 区追加（与现有 `from src.services.inference.base import ...` 同区）：
```python
from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.exceptions import NodeCancelled, NodeTimeout
```

然后在模块级 helpers 区（`vae_decode` 之后、`class DiffusersImageBackend` 之前，约 :256 那行 `# ---` 分隔线上方）插入：
```python
# --- within-node cancel (V1.5 Lane G / D14) --------------------------------
#
# spec §4.4 的关键机制：diffusers 的 callback_on_step_end 在 to_thread 工作
# 线程里、每个采样步之间被 invoke。它是唯一能让 cancel/timeout 信号穿过
# to_thread 边界、真正停掉 CUDA kernel 的钩子 —— asyncio.wait_for 单独用不行，
# 它取消的是 awaiting，to_thread 里的 CUDA kernel 照跑。
#
# _make_step_callback 是闭包工厂：每次 sample()/infer() 调用绑定一个**当次**的
# CancelFlag（不能模块级共享 —— spec §4.4 伪代码的模块级 cancel_flag 是简写）。


def _make_step_callback(cancel_flag: CancelFlag | None):
    """构造一个 diffusers callback_on_step_end 回调。

    diffusers 0.38 的回调签名: (pipe, step, timestep, callback_kwargs) -> dict。
    每个采样步之间被 invoke 一次。cancel_flag 为 None 时回调是 no-op（V1
    直调兼容路径，无取消能力）。
    """

    def _on_step_end(pipe, step, timestep, callback_kwargs):
        if cancel_flag is not None and cancel_flag.is_set():
            # 抛出会中断 diffusers 的扩散循环 → 停掉后续 CUDA kernel launch。
            raise NodeCancelled(cancel_flag.reason or "cancelled")
        return callback_kwargs

    return _on_step_end
```

- [ ] **Step 4: 改 `sample()` 接 callback**

`image_diffusers.py` 的 `sample()`（当前 :180-232），改函数签名加 `cancel_flag` kwarg，并把它接进 `pipe(...)` 调用。把 `def sample(...)` 整个函数替换为：
```python
def sample(
    pipe: Any,
    conditioning: dict[str, Any],
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Any | None = None,
    cancel_flag: CancelFlag | None = None,
    **kwargs: Any,
) -> Any:
    """Run the denoising loop and return an unpacked LATENT tensor (B,C,H,W).

    V1.5 Lane G: 接 diffusers callback_on_step_end —— 每采样步 check
    cancel_flag，命中则抛 NodeCancelled 中断扩散循环（停 CUDA kernel）。
    cancel_flag=None 时回调是 no-op，行为与 V1 一致。

    Implementation note: rather than copy diffusers' ~80-line denoising loop
    we ride pipe.__call__(output_type="latent"). That keeps us shielded from
    scheduler/quantization edge cases the pipeline already handles. The
    cost is one extra encode_prompt call inside __call__; we eat that and
    save the maintenance burden of duplicating the loop here.

    diffusers returns packed latents (B, H*W, C) for output_type="latent",
    which loses the spatial layout the VAE decode needs. We unpack here
    using ids reconstructed from the requested (height, width) — that
    matches the same arithmetic Flux2Pipeline.prepare_latents uses, so
    vae_decode below can stay agnostic of the original resolution.
    """
    import torch

    out = pipe(
        prompt_embeds=conditioning["prompt_embeds"],
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
        return_dict=False,
        callback_on_step_end=_make_step_callback(cancel_flag),
        **kwargs,
    )
    packed = out[0] if isinstance(out, tuple) else out

    # Reconstruct latent_ids the same way Flux2Pipeline.prepare_latents does:
    # round H,W to (vae_scale_factor*2) multiples, halve once for the
    # transformer's patchified shape.
    sf = pipe.vae_scale_factor
    latent_h = 2 * (height // (sf * 2))
    latent_w = 2 * (width // (sf * 2))
    batch_size = packed.shape[0]
    shape_proxy = torch.empty(
        (batch_size, 1, latent_h // 2, latent_w // 2),
        device=packed.device,
    )
    latent_ids = pipe._prepare_latent_ids(shape_proxy).to(packed.device)
    return pipe._unpack_latents_with_ids(packed, latent_ids)
```

> 注意：测试里的 `_FakePipe.__call__` 不返回需要 unpack 的真 latent —— `test_sample_runs_all_steps_when_not_cancelled` 用的 fake pipe 返回 `(["latent-after-..."],)`，`out[0]` 是个 list，后面 `.shape` 会 AttributeError。**修正 fake pipe**：让 `_FakePipe.__call__` 在 `callback_on_step_end` 跑完后直接返回一个带 `.shape` 的对象。把上面测试里的 `_FakePipe.__call__` return 改为返回一个最小 tensor stub —— 见 Step 5 的「测试修正」。

- [ ] **Step 5: 修正 fake pipe 让 `sample()` 的 unpack 路径不炸**

`sample()` 在 `pipe(...)` 之后还要做 `packed.shape` / `pipe._prepare_latent_ids` / `pipe._unpack_latents_with_ids`。fake pipe 必须提供这些。把 `test_image_adapter_cancel.py` 里的 `_FakePipe` 替换为：
```python
class _FakeLatent:
    """最小 latent stub：sample() 的 unpack 路径只读 .shape / .device。"""
    def __init__(self, batch=1):
        self.shape = (batch, 16, 32, 32)
        self.device = "cpu"


class _FakePipe:
    """模拟 diffusers pipe.__call__ 的采样循环：跑 num_inference_steps 步，
    每步 invoke callback_on_step_end。完全同步、纯 Python、不碰 CUDA。"""

    def __init__(self, step_sleep_s: float = 0.0):
        self.vae_scale_factor = 8
        self._step_sleep_s = step_sleep_s
        self.steps_run = 0

    def __call__(self, *, num_inference_steps, callback_on_step_end=None, **kw):
        import time
        for step in range(num_inference_steps):
            if callback_on_step_end is not None:
                callback_on_step_end(self, step, 1000 - step, {})
            self.steps_run += 1
            if self._step_sleep_s:
                time.sleep(self._step_sleep_s)
        return (_FakeLatent(),)

    # sample() 的 unpack 尾段需要这两个 —— 直接透传，本测试不验证 unpack 正确性
    def _prepare_latent_ids(self, shape_proxy):
        return shape_proxy

    def _unpack_latents_with_ids(self, packed, latent_ids):
        return packed
```
（这段替换掉 Step 1 测试文件里原来的 `_FakePipe`。`import torch` 在 `sample()` 内部 —— 测试环境 `CUDA_VISIBLE_DEVICES=""`，`torch.empty((1,1,16,16), device="cpu")` 走 CPU，不碰 GPU。若测试环境无 torch，则 `sample()` 的 unpack 段无法跑 —— 见 Step 7 的 fallback 说明。）

- [ ] **Step 6: 改 `infer()` —— 包 `wait_for(to_thread(...))`，cancel/timeout 共用 flag**

`image_diffusers.py` 的 `infer()`（当前 :561-617），改签名加 `cancel_flag` kwarg，把 `out = self._pipe(**call_kwargs)` 那段（:597-599）替换为 `wait_for + to_thread` 结构。把 `async def infer(...)` 整个方法替换为：
```python
    async def infer(
        self, req: InferenceRequest, cancel_flag: CancelFlag | None = None
    ) -> InferenceResult:
        """Run image generation.

        V1.5 Lane G: 采样跑在 asyncio.to_thread 里，外面包 asyncio.wait_for
        做超时；cancel_flag（runner pipe-reader 收 Abort 时 set）与超时分支
        共用同一个 flag —— callback_on_step_end 在采样步之间 check 它，命中
        就抛 NodeCancelled 停 CUDA kernel。

        cancel_flag=None 时内部自建一个（V1 直调路径无外部取消源，但超时
        路径仍需要一个 flag 来中断在飞的采样线程）。
        """
        if not isinstance(req, ImageRequest):
            raise TypeError(
                f"DiffusersImageBackend expects ImageRequest, got {type(req).__name__}"
            )
        if self._pipe is None:
            raise RuntimeError("DiffusersImageBackend.load() must be called before infer()")

        import inspect
        import secrets

        import torch

        # 外部没传 flag（V1 直调）时内部自建 —— 超时分支也要靠它中断采样线程。
        if cancel_flag is None:
            cancel_flag = CancelFlag()

        self._apply_loras(req.loras)

        # ComfyUI-style: when no seed supplied, draw a fresh 64-bit random one
        # so every run is reproducible (the seed is echoed back in metadata).
        seed = req.seed if req.seed is not None else secrets.randbelow(2**63)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Flux2KleinPipeline (and other distilled Flux variants) don't accept
        # negative_prompt — pass only kwargs the pipeline's __call__ declares.
        candidate_kwargs = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt or None,
            "width": req.width,
            "height": req.height,
            "num_inference_steps": req.steps,
            "guidance_scale": req.cfg_scale,
            "generator": generator,
        }
        accepted = set(inspect.signature(self._pipe.__call__).parameters.keys())
        call_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
        # diffusers 支持 callback_on_step_end 的 pipeline 才挂回调；fake / 老
        # pipeline 不声明该参数时跳过（within-node cancel 退化为边界 cancel）。
        if "callback_on_step_end" in accepted:
            call_kwargs["callback_on_step_end"] = _make_step_callback(cancel_flag)

        def _run_pipe():
            return self._pipe(**call_kwargs)

        t0 = time.monotonic()
        try:
            if req.timeout_s is not None:
                out = await asyncio.wait_for(
                    asyncio.to_thread(_run_pipe), timeout=req.timeout_s
                )
            else:
                out = await asyncio.to_thread(_run_pipe)
        except asyncio.TimeoutError:
            # wait_for 取消的是 awaiting；to_thread 里的采样线程还在跑 CUDA。
            # set flag → callback 在下一采样步抛 NodeCancelled → 线程自行退出。
            cancel_flag.set("node timeout")
            raise NodeTimeout(req.timeout_s)
        latency_ms = int((time.monotonic() - t0) * 1000)

        image = out.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return InferenceResult(
            media_type="image/png",
            data=png_bytes,
            metadata={
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "seed": seed,
                "loras": [{"name": s.name, "strength": s.strength} for s in req.loras],
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
```

> 关键点：`NodeCancelled`（外部 flag 被 set，callback 在采样线程里抛）会从 `_run_pipe` → `to_thread` → `await` 一路冒泡出 `infer()`，**不**被 `except asyncio.TimeoutError` 捕获 —— 正是我们要的：外部 cancel 抛 `NodeCancelled`，超时抛 `NodeTimeout`，两者可区分。`test_infer_external_cancel_flag_is_honored` 覆盖这条。

- [ ] **Step 7: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_image_adapter_cancel.py -v`
Expected: 7 个用例全 PASS（3 个 `_make_step_callback` 纯函数 + 4 个 fake pipe 集成）。

> fallback：若测试环境**未装 torch**（`image` extra 未装），`sample()` 内的 `import torch` 会 `ModuleNotFoundError`，3 个调 `sample()` 的集成用例会 error。此时在 `test_image_adapter_cancel.py` 顶部加：
> ```python
> torch = pytest.importorskip("torch")
> ```
> 4 个 `_make_step_callback` / `infer` 用例不依赖 `sample()` 的 unpack 段，但 `infer()` 也 `import torch`（`torch.Generator`）—— 所以 `importorskip` 放文件级，整文件随 torch 缺失而 skip。`_make_step_callback` 的 3 个纯函数用例不碰 torch，可拆到单独文件 `test_step_callback.py` 保证无 torch 也能跑。**实施时先跑一次 `python -c "import torch"` 探测**：装了就保持单文件，没装就拆 `_make_step_callback` 用例到 `test_step_callback.py` 并在 `test_image_adapter_cancel.py` 顶部加 `importorskip`。

- [ ] **Step 8: 跑 image adapter 既有 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q -k "image or diffusers"`
Expected: PASS。重点确认既有调 `infer()` / `sample()` 的用例不受新增 kwarg 影响（`cancel_flag` 有默认值 `None`，旧调用方不传照常工作；`sample()` 的 `callback_on_step_end` 对真 diffusers pipe 是合法 kwarg）。若某用例的 fake pipe 不接受 `callback_on_step_end` kwarg 而炸 —— 那是该测试的 fake pipe 需补一个 `**kw` 吞掉，记录但本 Lane 范围内修。

- [ ] **Step 9: Commit**

```bash
cd backend && git add src/services/inference/image_diffusers.py tests/test_image_adapter_cancel.py
git commit -m "feat(image): wire diffusers callback_on_step_end for within-node cancel

D14 / G2 critical-gap rewrite. sample() and infer() now accept a
CancelFlag; sample() builds a callback_on_step_end closure that raises
NodeCancelled between sampling steps when the flag is set — the only
mechanism that actually stops the CUDA kernel (asyncio.wait_for alone
cancels the awaiting, not the kernel). infer() wraps the pipe call in
wait_for(to_thread(...)); the timeout branch sets the SAME flag so the
in-flight sampler thread self-interrupts on its next step. External
cancel raises NodeCancelled, timeout raises NodeTimeout — distinguishable.
V1.5 Lane G, spec 4.4."
```

---

## Task 4: `QueuedTask` —— PriorityQueue 排序载体

spec §3.5 给的 `QueuedTask` 是 `@dataclass(order=True)`，`sort_key` 参与比较，`task_id` / `workflow_spec` 标 `compare=False`。这里先把它单独立起来 + 单测排序语义（spec §1.1：interactive=0 > batch=10，同级 `queued_at` FIFO），下一个 Task 才是 `GroupScheduler` 本体。

> 与 spec 偏差（见 plan 顶部注 1）：spec §3.5 写 `sort_key: tuple[int, datetime]`，§2.2 写三元组 `(priority, queued_at, task_id)`。本 plan 用三元组 —— 保证 sort_key 永远可全序比较，`PriorityQueue` 不会 fallback 到比 `QueuedTask` 的不可比字段。

**Files:**
- Create: `backend/src/services/scheduler/__init__.py`（空包标记）
- Create: `backend/src/services/scheduler/group_scheduler.py`（本 Task 只放 `QueuedTask` + 优先级常量；Task 5 加 `GroupScheduler`）
- Test: `backend/tests/test_group_scheduler.py`（新建，本 Task 先放 `QueuedTask` 用例）

- [ ] **Step 1: 写失败测试 —— QueuedTask 排序**

新建 `backend/tests/test_group_scheduler.py`：
```python
"""Lane G: QueuedTask + GroupScheduler 单元测试（纯内存，无 DB / GPU / runner）。"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.services.scheduler.group_scheduler import (
    PRIORITY_BATCH,
    PRIORITY_INTERACTIVE,
    QueuedTask,
)

_T0 = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _qt(task_id: int, priority: int, offset_s: float = 0.0) -> QueuedTask:
    return QueuedTask.create(
        task_id=task_id,
        priority=priority,
        queued_at=_T0 + timedelta(seconds=offset_s),
        workflow_spec={"id": task_id},
    )


def test_priority_constants():
    assert PRIORITY_INTERACTIVE == 0
    assert PRIORITY_BATCH == 10
    assert PRIORITY_INTERACTIVE < PRIORITY_BATCH


def test_interactive_sorts_before_batch():
    """优先级低的数字先出（interactive=0 在 batch=10 前）。"""
    interactive = _qt(1, PRIORITY_INTERACTIVE, offset_s=100)  # 即使晚入队
    batch = _qt(2, PRIORITY_BATCH, offset_s=0)  # 即使早入队
    assert interactive < batch


def test_same_priority_fifo_by_queued_at():
    """同优先级内，queued_at 早的先出（FIFO）。"""
    early = _qt(1, PRIORITY_BATCH, offset_s=0)
    late = _qt(2, PRIORITY_BATCH, offset_s=5)
    assert early < late


def test_same_priority_same_time_breaks_by_task_id():
    """同优先级 + 同 queued_at（精度内相等）时，task_id 兜底，sort_key 仍可全序。"""
    a = _qt(10, PRIORITY_BATCH, offset_s=0)
    b = _qt(20, PRIORITY_BATCH, offset_s=0)
    assert a < b  # task_id 10 < 20
    # 关键：比较不会因为 workflow_spec dict 不可比而 TypeError
    assert sorted([b, a])[0].task_id == 10


@pytest.mark.asyncio
async def test_queued_task_works_in_priority_queue():
    """放进真 asyncio.PriorityQueue，弹出顺序 = 优先级 + FIFO。"""
    q: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()
    await q.put(_qt(1, PRIORITY_BATCH, offset_s=0))
    await q.put(_qt(2, PRIORITY_INTERACTIVE, offset_s=10))  # 晚入队但高优先级
    await q.put(_qt(3, PRIORITY_BATCH, offset_s=1))
    order = []
    while not q.empty():
        order.append((await q.get()).task_id)
    assert order == [2, 1, 3]  # interactive 先, 然后 batch 内 FIFO
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_group_scheduler.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.scheduler'`

- [ ] **Step 3: 建 `scheduler` 包 + `QueuedTask`**

新建 `backend/src/services/scheduler/__init__.py`（空文件）：
```python
"""V1.5 主进程调度层 —— GroupScheduler + QueuedTask。"""
```

新建 `backend/src/services/scheduler/group_scheduler.py`（本 Task 部分；Task 5 追加 `GroupScheduler`）：
```python
"""GroupScheduler —— 主进程内每 GPU group 一个 asyncio.PriorityQueue 派发器。

spec §3.5 / §1.1 / §4.4。每个 image/TTS GPU group 一个 GroupScheduler 实例：
  * 一个 asyncio.PriorityQueue[QueuedTask] —— 2 级优先级 + 同级 FIFO
  * 一个 dispatcher loop —— 弹最小者 → 标 running → 调 executor → 终态回收
  * cancel_events: dict[task_id, asyncio.Event] —— 节点边界 cancel 的信号源
  * inflight_tasks: dict[task_id, asyncio.Task] —— 正在执行的 asyncio.Task 句柄

LLM group 没有 dispatch 队列（spec §1.2：LLM 直连 vLLM HTTP，不串行化），
所以 LLM group 不创建 GroupScheduler —— 本模块只服务 image/TTS group。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# spec §1.1：2 级优先级。数字小 = 优先级高（asyncio.PriorityQueue 弹最小者）。
PRIORITY_INTERACTIVE = 0
PRIORITY_BATCH = 10

# spec §4.7 case 4：某 group 队列堆积超过此值，enqueue 拒绝（路由层转 503）。
QUEUE_CAPACITY = 1000


@dataclass(order=True)
class QueuedTask:
    """PriorityQueue 里的一个待派发 task。

    只有 sort_key 参与比较（dataclass order=True 按字段顺序比，sort_key 是
    第一个且唯一 compare=True 的字段）。task_id / workflow_spec 标 compare=False
    —— workflow_spec 是 dict，不可比；不排除掉的话 PriorityQueue 在 sort_key
    相等时会 fallback 比它，直接 TypeError。

    sort_key = (priority, queued_at, task_id) 三元组（spec §2.2）。第三元
    task_id 保证全序：同 priority + 同 queued_at（datetime 精度内相等）时仍
    有确定顺序，PriorityQueue 永远不会 fallback 到比不可比字段。
    """

    sort_key: tuple[int, datetime, int]
    task_id: int = field(compare=False)
    workflow_spec: dict = field(compare=False)

    @classmethod
    def create(
        cls,
        *,
        task_id: int,
        priority: int,
        queued_at: datetime,
        workflow_spec: dict,
    ) -> "QueuedTask":
        """构造 QueuedTask，sort_key 由 (priority, queued_at, task_id) 拼成。"""
        return cls(
            sort_key=(priority, queued_at, task_id),
            task_id=task_id,
            workflow_spec=workflow_spec,
        )

    @property
    def priority(self) -> int:
        return self.sort_key[0]

    @property
    def queued_at(self) -> datetime:
        return self.sort_key[1]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_group_scheduler.py -v`
Expected: 5 个 `QueuedTask` 用例全 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/scheduler/__init__.py src/services/scheduler/group_scheduler.py tests/test_group_scheduler.py
git commit -m "feat(scheduler): add QueuedTask priority-queue ordering primitive

QueuedTask is a dataclass(order=True) whose sort_key (priority,
queued_at, task_id) is the only compare field — workflow_spec is a
dict and would TypeError if PriorityQueue fell back to comparing it.
Third key task_id guarantees a total order so the fallback never
happens (spec writes a 2-tuple in 3.5 but a 3-tuple in 2.2 — the
3-tuple is correct). V1.5 Lane G, spec 1.1/3.5."
```

---

## Task 5: `GroupScheduler` 本体 —— 派发 loop + cancel 双层 + 堆积 503

`GroupScheduler` 把队列、dispatcher loop、cancel、inflight 回收串起来。关键设计（含 plan 顶部注 2、3 的判断）：

- **executor 注入解耦** —— 构造时接收一个 `executor: Callable[[int, dict, asyncio.Event, CancelFlag], Awaitable[dict]]`。dispatcher 对每个 task 调它。真 executor（内部 dispatch 节点到 RunnerClient）由 Lane S 注入；本 Lane 单测注入 fake。
- **cancel 双层**：
  - 节点边界 cancel —— `request_cancel(task_id, reason)` set `cancel_events[task_id]`。dispatcher 弹出 task 时**先** check：若已 set，直接标 cancelled、不调 executor（plan 注 2：不从 PriorityQueue 物理删除，弹出时跳过）。executor 内部（Lane S）每个节点 dispatch 前也 check 同一个 event —— 那是 Lane S 的事，本 Lane 把 event 传进 executor 即可。
  - within-node cancel —— `request_cancel` 同时 set 该 task 的 `CancelFlag`（每个 inflight task 一个）。CancelFlag 被传进 executor，executor 再传给 adapter（Task 3 的 `infer(req, cancel_flag=...)`）。
- **堆积 503** —— `enqueue` 时 `queue.qsize() >= QUEUE_CAPACITY` 抛 `QueueFullError`。
- **inflight 回收** —— dispatcher 为每个 task `asyncio.create_task(self._run_one(...))`，记进 `inflight_tasks`；task 终态（完成/失败/取消）后从 `inflight_tasks` / `cancel_events` / `cancel_flags` 清理。

**Files:**
- Modify: `backend/src/services/scheduler/group_scheduler.py`（追加 `GroupScheduler`）
- Test: `backend/tests/test_group_scheduler.py`（追加 `GroupScheduler` 用例）

- [ ] **Step 1: 写失败测试 —— GroupScheduler 全部行为**

在 `backend/tests/test_group_scheduler.py` 末尾追加：
```python
from src.services.inference.exceptions import NodeCancelled, QueueFullError
from src.services.scheduler.group_scheduler import (
    QUEUE_CAPACITY,
    GroupScheduler,
)


def _make_scheduler(executor):
    return GroupScheduler(group_id="image", executor=executor)


@pytest.mark.asyncio
async def test_enqueue_dispatch_completes():
    """enqueue → dispatcher 弹出 → executor 跑 → task 进 results。"""
    results = {}

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        results[task_id] = {"ok": True, "spec": spec}
        return results[task_id]

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(
        task_id=1, priority=PRIORITY_INTERACTIVE,
        queued_at=_T0, workflow_spec={"id": 1},
    )
    await sched.join()  # 等队列里的 task 全部派发完
    await sched.stop()
    assert results == {1: {"ok": True, "spec": {"id": 1}}}


@pytest.mark.asyncio
async def test_priority_order_interactive_first():
    """batch task 先入队，interactive 后入队 —— interactive 先被 executor 跑。"""
    run_order = []

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        run_order.append(task_id)
        return {}

    sched = _make_scheduler(fake_executor)
    # 先把两个 task 灌进队列再 start —— 保证 dispatcher 启动时队列里已有 2 个，
    # 排序才有意义（否则先 enqueue 的可能在第二个 enqueue 前就被弹走了）。
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.enqueue(task_id=2, priority=PRIORITY_INTERACTIVE,
                        queued_at=_T0, workflow_spec={})
    await sched.start()
    await sched.join()
    await sched.stop()
    assert run_order == [2, 1]  # interactive 先


@pytest.mark.asyncio
async def test_cancel_while_queued_skips_executor():
    """task 还在排队（未 dispatch）时 cancel —— dispatcher 弹出时跳过，
    executor 根本不被调用，task 标 cancelled。"""
    executor_calls = []

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        executor_calls.append(task_id)
        return {}

    sched = _make_scheduler(fake_executor)
    # 不 start —— 先 enqueue + cancel，再 start，保证 cancel 发生在 dispatch 前
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    sched.request_cancel(1, reason="user changed mind")
    await sched.start()
    await sched.join()
    await sched.stop()
    assert executor_calls == []  # executor 从未被调
    assert sched.get_status(1) == "cancelled"


@pytest.mark.asyncio
async def test_cancel_inflight_sets_cancel_flag():
    """task 正在执行时 cancel —— cancel_event 和 CancelFlag 都被 set，
    executor 能观察到 → 抛 NodeCancelled → task 标 cancelled。"""
    started = asyncio.Event()

    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        started.set()
        # 模拟一个长节点：轮询 cancel_flag（adapter 实际是 callback 里 check）
        for _ in range(100):
            await asyncio.sleep(0.01)
            if cancel_flag.is_set():
                raise NodeCancelled(cancel_flag.reason or "cancelled")
        return {}

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_INTERACTIVE,
                        queued_at=_T0, workflow_spec={})
    await started.wait()  # 等 executor 真的开始跑
    sched.request_cancel(1, reason="abort inflight")
    await sched.join()
    await sched.stop()
    assert sched.get_status(1) == "cancelled"
    assert sched.get_cancel_reason(1) == "abort inflight"


@pytest.mark.asyncio
async def test_executor_exception_marks_failed():
    """executor 抛非 cancel 异常 —— task 标 failed，dispatcher 不挂、继续服务。"""
    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        raise RuntimeError("node OOM")

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    # dispatcher 仍活着 —— 再灌一个能成功的
    ok = {}

    async def ok_executor(task_id, spec, cancel_event, cancel_flag):
        ok[task_id] = True
        return {}

    sched._executor = ok_executor  # 换 executor 验证 loop 没死
    await sched.enqueue(task_id=2, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    await sched.stop()
    assert sched.get_status(1) == "failed"
    assert sched.get_status(2) == "completed"


@pytest.mark.asyncio
async def test_enqueue_raises_queue_full_on_backlog():
    """队列堆积到 QUEUE_CAPACITY 后，enqueue 抛 QueueFullError。"""
    async def slow_executor(task_id, spec, cancel_event, cancel_flag):
        await asyncio.sleep(60)  # 永远跑不完 —— 让队列堆起来
        return {}

    sched = _make_scheduler(slow_executor)
    # 不 start dispatcher —— 队列只进不出，直接灌满
    for i in range(QUEUE_CAPACITY):
        await sched.enqueue(task_id=i, priority=PRIORITY_BATCH,
                            queued_at=_T0, workflow_spec={})
    with pytest.raises(QueueFullError) as ei:
        await sched.enqueue(task_id=99999, priority=PRIORITY_BATCH,
                            queued_at=_T0, workflow_spec={})
    assert ei.value.group_id == "image"
    assert ei.value.capacity == QUEUE_CAPACITY


@pytest.mark.asyncio
async def test_inflight_cleared_after_completion():
    """task 终态后从 inflight_tasks / cancel_events 清理，不泄漏。"""
    async def fake_executor(task_id, spec, cancel_event, cancel_flag):
        return {}

    sched = _make_scheduler(fake_executor)
    await sched.start()
    await sched.enqueue(task_id=1, priority=PRIORITY_BATCH,
                        queued_at=_T0, workflow_spec={})
    await sched.join()
    await sched.stop()
    assert 1 not in sched.inflight_tasks
    assert 1 not in sched.cancel_events
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_group_scheduler.py -v`
Expected: 新追加的 7 个用例 FAIL（`ImportError: cannot import name 'GroupScheduler'`），Task 4 的 5 个 `QueuedTask` 用例仍 PASS。

- [ ] **Step 3: 实现 `GroupScheduler`**

在 `backend/src/services/scheduler/group_scheduler.py` 末尾追加。先在文件顶部 import 区补：
```python
import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.exceptions import NodeCancelled, QueueFullError

logger = logging.getLogger(__name__)
```
（`from __future__ import annotations` / `from dataclasses import ...` / `from datetime import datetime` 已在 Task 4 加过 —— 把这些 import 合并到文件顶部一个 import 区，不要重复。）

然后在 `QueuedTask` 类之后追加：
```python
# executor 回调签名：(task_id, workflow_spec, cancel_event, cancel_flag) -> result dict。
# 真 executor 由 Lane S 注入（内部把 dispatch 节点投给 RunnerClient）；本 Lane
# 单测注入 fake。cancel_event 给节点边界 cancel（executor 每节点 dispatch 前
# check）；cancel_flag 给 within-node cancel（executor 传给 adapter.infer）。
ExecutorCallable = Callable[
    [int, dict, asyncio.Event, CancelFlag], Awaitable[dict]
]


class GroupScheduler:
    """一个 GPU group 的派发器：PriorityQueue + dispatcher loop + cancel 双层。

    生命周期：start() 起 dispatcher loop → enqueue() 投 task → dispatcher 弹出
    并为每个 task 起一个 _run_one asyncio.Task → join() 等队列排空 → stop() 关
    dispatcher。

    不接 DB / RunnerClient / TaskRingBuffer —— 纯内存逻辑，executor 注入解耦
    （见 ExecutorCallable）。真接线（DB 持久化、节点 dispatch、ring buffer
    推送）由 Lane S/I 完成。
    """

    def __init__(
        self,
        group_id: str,
        executor: ExecutorCallable,
        *,
        capacity: int = QUEUE_CAPACITY,
    ) -> None:
        self.group_id = group_id
        self._executor = executor
        self._capacity = capacity
        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()
        self.cancel_events: dict[int, asyncio.Event] = {}
        self.cancel_flags: dict[int, CancelFlag] = {}
        self.inflight_tasks: dict[int, asyncio.Task] = {}
        self._status: dict[int, str] = {}          # task_id -> queued/running/completed/failed/cancelled
        self._cancel_reason: dict[int, str] = {}   # task_id -> reason
        self._dispatcher: asyncio.Task | None = None
        self._stopping = False

    # ---- 生命周期 --------------------------------------------------------

    async def start(self) -> None:
        """起 dispatcher loop。幂等：已在跑则 no-op。"""
        if self._dispatcher is None or self._dispatcher.done():
            self._stopping = False
            self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """停 dispatcher loop。已 inflight 的 task 不强杀（等它们自然终态）。"""
        self._stopping = True
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except asyncio.CancelledError:
                pass
            self._dispatcher = None
        # 等所有 inflight task 收尾（终态 handler 会自行清理 dict）
        if self.inflight_tasks:
            await asyncio.gather(
                *self.inflight_tasks.values(), return_exceptions=True
            )

    async def join(self) -> None:
        """等队列里的 task 全部派发完 + 所有 inflight task 终态。"""
        await self._queue.join()
        if self.inflight_tasks:
            await asyncio.gather(
                *list(self.inflight_tasks.values()), return_exceptions=True
            )

    # ---- 入队 ------------------------------------------------------------

    async def enqueue(
        self,
        *,
        task_id: int,
        priority: int,
        queued_at: datetime,
        workflow_spec: dict,
    ) -> None:
        """投一个 task 进队列。队列堆积超过 capacity 时抛 QueueFullError。"""
        if self._queue.qsize() >= self._capacity:
            raise QueueFullError(self.group_id, self._capacity)
        qt = QueuedTask.create(
            task_id=task_id,
            priority=priority,
            queued_at=queued_at,
            workflow_spec=workflow_spec,
        )
        # cancel_event / cancel_flag 在入队时就建好 —— 这样 task 还在排队时
        # request_cancel 也有东西可 set（dispatcher 弹出时会 check event）。
        self.cancel_events.setdefault(task_id, asyncio.Event())
        self.cancel_flags.setdefault(task_id, CancelFlag())
        self._status[task_id] = "queued"
        await self._queue.put(qt)

    # ---- cancel 双层 -----------------------------------------------------

    def request_cancel(self, task_id: int, reason: str = "cancelled") -> bool:
        """请求取消一个 task。同时 set 节点边界 cancel 的 asyncio.Event 和
        within-node cancel 的 CancelFlag —— 两层信号一次发齐。

        返回 task 是否已知（在排队 / 执行中）。已终态的 task 返回 False。
        """
        if task_id not in self.cancel_events:
            return False
        if self._status.get(task_id) in ("completed", "failed", "cancelled"):
            return False
        self._cancel_reason[task_id] = reason
        self.cancel_events[task_id].set()       # 节点边界 cancel
        self.cancel_flags[task_id].set(reason)  # within-node cancel（穿 to_thread）
        return True

    # ---- dispatcher ------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """弹出 PriorityQueue 最小者，为每个 task 起一个 _run_one task。"""
        while not self._stopping:
            qt = await self._queue.get()
            try:
                # plan 注 2：cancel 还在排队的 task 不从 PriorityQueue 物理删除，
                # 弹出时 check —— 已 cancel 则直接标 cancelled，不调 executor。
                cancel_event = self.cancel_events.get(qt.task_id)
                if cancel_event is not None and cancel_event.is_set():
                    self._finalize(
                        qt.task_id, "cancelled",
                        self._cancel_reason.get(qt.task_id, "cancelled"),
                    )
                    continue
                self._status[qt.task_id] = "running"
                t = asyncio.create_task(self._run_one(qt))
                self.inflight_tasks[qt.task_id] = t
            finally:
                self._queue.task_done()

    async def _run_one(self, qt: QueuedTask) -> None:
        """跑单个 task：调 executor，按结果 / 异常落终态。"""
        task_id = qt.task_id
        cancel_event = self.cancel_events[task_id]
        cancel_flag = self.cancel_flags[task_id]
        try:
            await self._executor(
                task_id, qt.workflow_spec, cancel_event, cancel_flag
            )
            self._finalize(task_id, "completed", None)
        except NodeCancelled as e:
            self._finalize(task_id, "cancelled", e.reason)
        except asyncio.CancelledError:
            # dispatcher stop() 取消了我们 —— 不当 failed，标 cancelled。
            self._finalize(task_id, "cancelled", "scheduler stopped")
            raise
        except Exception as e:
            logger.error(
                "group %s task %d failed: %s", self.group_id, task_id, e
            )
            self._finalize(task_id, "failed", str(e))

    def _finalize(self, task_id: int, status: str, reason: str | None) -> None:
        """落终态 + 清理 inflight / cancel 字典，防泄漏。"""
        self._status[task_id] = status
        if reason is not None:
            self._cancel_reason[task_id] = reason
        self.inflight_tasks.pop(task_id, None)
        self.cancel_events.pop(task_id, None)
        self.cancel_flags.pop(task_id, None)

    # ---- 查询（给单测 / 后续 Lane 的 observability 用） -------------------

    def get_status(self, task_id: int) -> str | None:
        return self._status.get(task_id)

    def get_cancel_reason(self, task_id: int) -> str | None:
        return self._cancel_reason.get(task_id)

    def queue_size(self) -> int:
        return self._queue.qsize()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_group_scheduler.py -v`
Expected: 全部 12 个用例 PASS（Task 4 的 5 个 `QueuedTask` + 本 Task 的 7 个 `GroupScheduler`）。

> 注意：`test_cancel_inflight_sets_cancel_flag` 依赖 `request_cancel` set `cancel_flag` 后 executor 的轮询 loop 能看到 —— fake executor 是 `await asyncio.sleep(0.01)` 轮询 `cancel_flag.is_set()`，真 adapter 是 diffusers callback 每步 check。两者都靠 `CancelFlag.is_set()` 跨边界可见。`_finalize` 在 `cancel_flags.pop` 之前 executor 已经 raise 完了，无竞态。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/scheduler/group_scheduler.py tests/test_group_scheduler.py
git commit -m "feat(scheduler): add GroupScheduler — per-group dispatch + cancel 2-tier

PriorityQueue dispatcher loop, executor injected (decoupled from
RunnerClient — Lane C/S wire the real one). Cancel 2-tier: request_cancel
sets both the node-boundary asyncio.Event and the within-node CancelFlag.
Cancel-while-queued is handled by skipping at dispatch pop (no physical
PriorityQueue removal — spec is silent, this matches task_queue.py's old
_cancelled pattern). enqueue raises QueueFullError at capacity. inflight
dicts cleaned on finalize. V1.5 Lane G, spec 3.5/4.4/4.7."
```

---

## Task 6: 注册 pytest markers + Lane G 整合验证 + lint 预检

spec §5.6 要求 `pytest.ini` 注册 `e2e` / `integration` / `chaos` markers。本 Lane 全是 unit 测试、不打 marker，但 Lane J 之前的每个 Lane 都可能写 `-m integration` 测试 —— 现在把 markers 注册掉，避免后续 `pytest -m` 报 `PytestUnknownMarkWarning`。这是顺手做的基建小事，放本 Lane（第一个写大量调度测试的 Lane）合适。

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: 注册 markers**

`backend/pyproject.toml` 的 `[tool.pytest.ini_options]` 段（当前有 `asyncio_mode = "auto"` / `testpaths = ["tests"]`），追加 `markers`：
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: 多进程 / mock runner subprocess 测试（CI 默认跑，spec §5.3）",
    "e2e: 真 GPU 测试（dev box 手动跑，CI skip，spec §5.4）",
    "chaos: 故障注入测试（每周手动跑，spec §5.5）",
]
```
（若 `[tool.pytest.ini_options]` 段还有其它 key，保留，只追加 `markers` list。）

- [ ] **Step 2: 确认 marker 注册生效**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest --markers | grep -E "integration|e2e|chaos"`
Expected: 看到三个 marker 的描述行（`@pytest.mark.integration: ...` 等）。

- [ ] **Step 3: Lane G 全部新测试 green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_cancel_flag.py tests/test_image_adapter_cancel.py tests/test_group_scheduler.py -v`
Expected: 6（cancel_flag）+ 7（image_adapter_cancel，若拆了 `_make_step_callback` 则 4 + 3）+ 12（group_scheduler）= 25 个用例全 PASS。

> 若 Task 3 Step 7 因无 torch 拆出了 `test_step_callback.py`，把它一并加进命令：`... tests/test_step_callback.py ...`。

- [ ] **Step 4: 后端全 suite 无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。新增 ~25 个用例，无 collection error、无 import error。重点确认：
- 既有 image adapter 测试不受 `infer()` / `sample()` 新增 `cancel_flag` kwarg 影响（默认 `None`，旧调用方不传照常）。
- 无 `PytestUnknownMarkWarning`（Step 1 已注册 markers）。

- [ ] **Step 5: lint 预检（push 前本地跑）**

Run: `cd backend && ruff check src/services/inference/exceptions.py src/services/inference/cancel_flag.py src/services/inference/image_diffusers.py src/services/scheduler/ tests/test_cancel_flag.py tests/test_image_adapter_cancel.py tests/test_group_scheduler.py`
Expected: 无 lint 错误。

- [ ] **Step 6: 开 PR**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git push -u origin <lane-G-branch>
gh pr create --title "feat: V1.5 Lane G — GroupScheduler + cancel 2-tier + image adapter rewrite" --body "$(cat <<'EOF'
## Summary
- `CancelFlag` —— cancel/timeout 信号穿 `asyncio.to_thread` 边界的唯一载体（`threading.Event` 薄包装 + reason）
- **image adapter 重写（D14 / G2 critical gap）** —— `sample()` / `infer()` 接 diffusers `callback_on_step_end`，每采样步 check `CancelFlag`，命中抛 `NodeCancelled` 停 CUDA kernel；`infer()` 包 `wait_for(to_thread(...))`，超时分支 set 同一 flag
- `QueuedTask` + `GroupScheduler` —— 每 GPU group 一个 `asyncio.PriorityQueue` 派发器，2 级优先级（interactive=0 / batch=10）+ 同级 `queued_at` FIFO；cancel 双层（节点边界 `asyncio.Event` + within-node `CancelFlag`）；队列堆积 `QueueFullError`
- 注册 pytest markers（`integration` / `e2e` / `chaos`，spec §5.6）

## 与 spec 的偏差 / 模糊处（已 flag）
- `QueuedTask.sort_key` 用 3 元组 `(priority, queued_at, task_id)` —— spec §3.5 写 2 元组、§2.2 写 3 元组，3 元组保证全序可比
- cancel 排队中的 task 不从 `PriorityQueue` 物理删除，dispatcher 弹出时 check 跳过 —— spec 未明说
- `GroupScheduler` 用注入的 executor 回调解耦 `RunnerClient`（Lane C 未落地），真接线属 Lane S

## Test plan
- [ ] `test_cancel_flag.py` green（set/is_set/clear/跨线程可见）
- [ ] `test_image_adapter_cancel.py` green（fake pipe：正常完成 / within-node cancel / timeout / 外部 flag）
- [ ] `test_group_scheduler.py` green（优先级 / FIFO / cancel 排队中 / cancel inflight / 堆积 503 / inflight 回收）
- [ ] 后端全 suite green，既有 image adapter 测试无回归
- [ ] E2E（dev box 手动，真 GPU，Lane S 接线后）：30 step sampler 第 5 步 cancel → <500ms 停 CUDA kernel
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneG-groupscheduler-cancel`。）

---

## Self-Review

**Spec 覆盖检查：** Lane G 在 spec「实施分 Lane」表里的职责是「GroupScheduler + priority + cancel 双层 + adapter 重写（image adapter 接 diffusers `callback_on_step_end`，cancel/timeout 走同一 `threading.Event` 穿过 to_thread，D14）。依赖：C」。

- **GroupScheduler + priority** → Task 4（`QueuedTask` 排序载体，2 级优先级 + FIFO，spec §1.1/§3.5）+ Task 5（`GroupScheduler` 本体：PriorityQueue 派发 loop + 堆积 503，spec §3.5/§4.7 case 4）。`QueuedTask` 是 `@dataclass(order=True)` + `sort_key` + `compare=False` —— 逐字对齐 spec §3.5 代码块（除 sort_key 元数见下偏差）。`GroupScheduler` 的 `cancel_events` / `inflight_tasks` 字段名对齐 spec §3.5 dataclass。
- **cancel 双层** → Task 5 `request_cancel` 一次 set 两层：节点边界 `cancel_events[task_id]`（spec §4.4「节点边界」）+ within-node `cancel_flags[task_id]`（spec §4.4「within-node」）。executor 收到 `cancel_event` + `cancel_flag` 两个参数 —— 节点边界 check 由 executor（Lane S）做，within-node 由 adapter（Task 3）做。
- **adapter 重写（D14）** → Task 2（`CancelFlag` —— spec §4.4 的 `threading.Event` 载体）+ Task 3（`image_diffusers.py` 的 `sample()` / `infer()` 接 `callback_on_step_end`）。Task 3 逐一落实 spec §4.4 三条关键性质：(a) `callback_on_step_end` 是唯一能穿 `to_thread` 停 CUDA kernel 的钩子 —— `_make_step_callback` 闭包每步 check；(b) cancel 与 timeout 走同一 flag —— `infer()` 的 `except asyncio.TimeoutError` 分支 `cancel_flag.set("node timeout")`，与外部 cancel 用同一个 `cancel_flag`；(c) `asyncio.wait_for` 单独用不行 —— `infer()` 注释明确，超时分支不杀线程、靠 callback 自行中断。
- **依赖：C** → **偏差**：Lane C plan 未写、`RunnerClient` 不存在。本 Lane 用注入的 `executor` 回调解耦，不 import `RunnerClient`，可独立实现 + 独立测试。真 executor（内部调 `RunnerClient.run_node` + dispatch 节点）由 Lane S 注入。已在 plan 顶部「偏差注 3」+ Architecture「Lane G 不做」明确。

**与 spec 的偏差 / 模糊处（已在 plan 顶部 + 对应 Task + commit message 显式标注）：**
1. **`QueuedTask.sort_key` 元数歧义** —— spec §3.5 写 `tuple[int, datetime]`（2 元），§2.2 写 `(priority, queued_at, task_id)`（3 元）。判断：用 3 元组。理由：纯 2 元组在同 priority + 同 `queued_at` 时，`PriorityQueue` 会 fallback 比 `QueuedTask` 的下一个 compare 字段，而 `workflow_spec` dict 不可比 → `TypeError`。3 元组（加 `task_id`）保证 sort_key 永远全序可比。`test_same_priority_same_time_breaks_by_task_id` 专门覆盖这条。已在 Task 4 实现 + commit message + plan 顶部注 1。
2. **cancel 排队中的 task 怎么从 `PriorityQueue` 摘** —— `asyncio.PriorityQueue` 不暴露按 key 删除。spec §4.7 case 1 只讲了「已 dispatch 但 runner 没收到」的情况。判断：不物理删除，dispatcher 弹出时 check `cancel_event.is_set()` 跳过（与 `task_queue.py` 旧实现的 `_cancelled` 标志同思路）。`test_cancel_while_queued_skips_executor` 覆盖。已在 Task 5 实现 + commit message + plan 顶部注 2。
3. **Lane C / `RunnerClient` 未落地** —— 见上「依赖：C」。executor 注入解耦。plan 顶部注 3。
4. **spec §4.4 的 `_on_step_end` 闭包捕获模块级 `cancel_flag`** 是伪代码简写，真实现必须 per-call 绑定。判断：用闭包工厂 `_make_step_callback(cancel_flag)`，每次 `sample()` / `infer()` 一个新 flag。已在 Task 3 实现 + plan 顶部注 4。

**spec 模糊处的其它判断：**
- spec §3.5 `GroupScheduler` 是 `@dataclass`，但本 Lane 实现为普通 class —— `GroupScheduler` 有 `start` / `stop` / `_dispatch_loop` 等大量行为方法 + 可变内部状态，`@dataclass` 只是省 `__init__` 样板，对一个有生命周期的服务类收益小、反而要处理 `field(default_factory=...)` 一堆。判断：普通 class，`__init__` 手写。字段名（`group_id` / `cancel_events` / `inflight_tasks`）仍对齐 spec §3.5。
- spec §3.5 `GroupScheduler` 有 `runner_client: RunnerClient | None` 字段 —— 本 Lane 不要这个字段（用 executor 注入替代，见偏差 3）。Lane S 接线时若仍想保留 `runner_client` 句柄可加，但本 Lane scope 内不需要。
- `infer()` 的 timeout 来源 —— spec §4.4 伪代码用 `timeout=T` 没说 T 哪来。判断：用 `req.timeout_s`（`InferenceRequest` 基类已有 `timeout_s: float | None`，见 `base.py:50`），`None` 时不包 `wait_for`（无超时，纯靠外部 cancel）。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有 exception / `CancelFlag` / adapter 改写 / `QueuedTask` / `GroupScheduler` 代码均完整给出。每个 Task 是「写失败测试 → 跑确认失败 → 最小实现 → 跑确认通过 → commit」闭环，命令带预期输出。唯一的条件分支是 Task 3 Step 7 的「装了 torch / 没装 torch」—— 给了明确的探测命令 + 两条路径，不是 placeholder。

**类型一致性：**
- `_make_step_callback(cancel_flag: CancelFlag | None)` 返回的回调签名 `(pipe, step, timestep, callback_kwargs) -> dict` 对齐 diffusers 0.38 `callback_on_step_end` 契约；fake pipe 的 `__call__` 按同签名 invoke。
- `sample(..., cancel_flag: CancelFlag | None = None, ...)` 与 `infer(self, req, cancel_flag: CancelFlag | None = None)` 的 `cancel_flag` 类型一致，默认 `None`（V1 兼容）。
- `ExecutorCallable = Callable[[int, dict, asyncio.Event, CancelFlag], Awaitable[dict]]` 与 `GroupScheduler._run_one` 里 `await self._executor(task_id, qt.workflow_spec, cancel_event, cancel_flag)` 的实参类型逐一对应（`task_id: int` / `workflow_spec: dict` / `cancel_event: asyncio.Event` / `cancel_flag: CancelFlag`）；fake executor 测试桩签名一致。
- `QueuedTask.sort_key: tuple[int, datetime, int]` 与 `create()` 里 `(priority, queued_at, task_id)` 的元素类型对应（`priority: int` / `queued_at: datetime` / `task_id: int`）。
- `QueueFullError(self.group_id, self._capacity)` 与 `exceptions.py` 里 `QueueFullError.__init__(group_id, capacity, retry_after_s=30)` 签名一致；`NodeTimeout(req.timeout_s)` 与 `NodeTimeout.__init__(timeout_s, reason=...)` 一致；`NodeCancelled(cancel_flag.reason or "cancelled")` 与 `NodeCancelled.__init__(reason="cancelled")` 一致。

**已知风险：**
- **adapter 重写依赖每个 image pipeline 都支持 `callback_on_step_end`** —— spec GSTACK REVIEW G2 明确标注「遗漏一个就退化为不可 cancel」。缓解：Task 3 Step 6 的 `infer()` 用 `inspect.signature` 探测 `"callback_on_step_end" in accepted`，不支持的 pipeline 跳过挂回调（within-node cancel 退化为边界 cancel，不报错）。真验证靠 spec §5.4 的 E2E 测试（真 GPU，30 step sampler 第 5 步 cancel）—— 本 Lane 的 fake pipe 测试只能验证「机制接对了」，验证不了「每个真 pipeline 都吃这个 kwarg」。Lane S 接线 + E2E 时需逐 pipeline 过。已在 PR body Test plan 列出 E2E 项。
- **`callback_on_step_end` 抛异常能否真的中断 diffusers 扩散循环** —— 本 Lane 的 fake pipe 测试假设「callback 抛 → 循环中断」，这是 diffusers 0.38 的行为（callback 在 `for step in timesteps` 循环体里被直接调用，异常自然冒泡终止循环）。但真 diffusers 是否在某些 pipeline 里 try/except 吞掉 callback 异常 —— 未在本 Lane 验证（无 GPU）。缓解：E2E 测试覆盖。若 E2E 发现某 pipeline 吞异常，fallback 是改用 `callback_kwargs` 里塞一个 sentinel 让 pipeline 自己 break（更 hacky，本 Lane 不预先实现）。
- **`_run_one` 的 `asyncio.CancelledError` 处理** —— `stop()` cancel dispatcher 后 `gather(inflight_tasks)`，inflight 的 `_run_one` 收到 `CancelledError` 会标 `cancelled` 并 re-raise。`_finalize` 在 re-raise 前已执行（清理 dict）。`gather(..., return_exceptions=True)` 吞掉 re-raise 的 `CancelledError`,不会泄漏。已在 Task 5 实现 + `test_inflight_cleared_after_completion` 间接覆盖（虽然该用例走的是正常完成路径，stop 路径的清理由 `_finalize` 同一份代码保证）。
- **`GroupScheduler` 不加锁** —— 设计为主进程 asyncio 单线程使用（与 `TaskRingBuffer` 同约束）。`request_cancel` 是同步方法、从 event loop 内调用（路由 handler）；`CancelFlag.set` 内部有 `threading.Lock` 保护跨线程写。`cancel_events` / `inflight_tasks` 等 dict 的读写全在 event loop 单线程内，无需额外锁。后续 Lane 若引入线程桥（spec §3.3 pipe-reader 读线程）接 `GroupScheduler`，接线方需注意 —— 但那是 Lane C/S 的 scope。
- **timeout 后在飞的采样线程** —— `infer()` 超时抛 `NodeTimeout` 后，`to_thread` 的工作线程还在跑（直到下一采样步 callback check flag 才退出）。最坏情况 = 一个采样步的时长（通常 <1s）线程仍占 GPU。这是 spec §4.4 设计本身的特性（不强杀线程），不是本 Lane 引入的缺陷。`test_infer_timeout_sets_flag_and_raises_node_timeout` 用 `step_sleep_s=0.05` 验证线程最终会因 flag 退出、不挂死。
