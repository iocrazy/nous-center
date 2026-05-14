# V1.5 Lane C: RunnerSupervisor + image/TTS Runner 子进程框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭出 V1.5 GPU Runner 的子进程框架——把 image/TTS 推理从 API 主进程挪进每个 GPU group 一个的 runner 子进程。本 Lane 用 **fake adapter**（不碰真 GPU / 真模型）跑通：msgpack IPC 协议、`multiprocessing.Pipe` 的 asyncio 桥接（F1 约束）、runner 内 pipe-reader + node-executor 双 asyncio task（D9）、RunnerSupervisor 的 spawn / watchdog / crash 检测 / backoff 重启（spec §4.2）、以及重启前的 GPU-free gate（F2）。**不**接 GroupScheduler、**不**迁真 ModelManager、**不**做 LLM runner——那些是 Lane D/E/G。

**Architecture:** 五块交付物，自底向上：
1. **msgpack 依赖 + IPC 协议层**（`src/runner/protocol.py`）—— spec §3.3 的 10 个消息类型（LoadModel/UnloadModel/RunNode/Abort/Ping + NodeResult/NodeProgress/ModelEvent/Pong/Ready），msgpack 编解码，`NOUS_IPC_FORMAT=json` dev fallback。
2. **PipeChannel**（`src/runner/pipe_channel.py`）—— 封装 F1 约束：读侧用 `loop.connect_read_pipe` 把 pipe fd 注册进 event loop（拿不到 fd 时退化为读线程桥到 `asyncio.Queue`）；写侧用专用写线程 + `queue.Queue`，对 `Pipe.send` 加 5s 超时（`Pipe.send` 本身无 timeout 参数）。
3. **fake adapter**（`src/runner/fake_adapter.py`）—— 实现 `InferenceAdapter` ABC，`load` / `infer` 全是可让出的 `await asyncio.sleep`，可配置 crash / slow / fail-load / 多 step 进度，给 IPC + 生命周期测试用，零 GPU 零模型。
4. **runner 子进程骨架**（`src/runner/runner_process.py`）—— `runner_main(group_id, gpus, conn)` 子进程入口：起独立 event loop，跑 pipe-reader task（持续读消息、RunNode 入内部队列、Abort 置 `threading.Event`）+ node-executor task（从队列取 RunNode、调 fake adapter、发 NodeProgress/NodeResult）。
5. **RunnerClient + RunnerSupervisor**（`src/runner/client.py` + `src/runner/supervisor.py`）—— 主进程侧：RunnerClient 是 PipeChannel 之上的节点级 RPC（`run_node` / `load_model` / `ping`）；RunnerSupervisor 负责 `multiprocessing.Process` 的 fork、watchdog（ping 超时检测）、crash 时 inflight task 标 failed、`RESTART_BACKOFF` 指数退避重启、F2 GPU-free gate。

**Tech Stack:** Python 3.12 / `multiprocessing`（`spawn` context，与 CUDA 子进程惯例一致）/ `msgpack`（本 Lane Task 1 新增依赖）/ `asyncio`（`loop.connect_read_pipe` / `run_in_executor` 写线程）/ `threading.Event`（跨 to_thread cancel 信号）/ pytest（`asyncio_mode = "auto"`）。

> **注意 — 与 spec / 简报的偏差（已核实，须知会）：**
>
> 1. **新代码落位 `backend/src/runner/`（新建目录），不是 `backend/src/workers/`。** 简报已点明：`src/workers/` 是既有的 engine 实现模块（`image_worker.py` / `tts_worker.py` / `celery_app.py` / `llm_engines/` / `tts_engines/`，celery worker 体系），spec 把新概念命名为「GPU Runner」正是为了避开与 `workers` 撞名。本 Lane 全部新模块进 `src/runner/`，与 `src/workers/` 物理隔离，零交叉 import。
>
> 2. **msgpack 不是现有依赖。** `backend/pyproject.toml` 的 `dependencies` 里没有 `msgpack`（已 grep 确认）。Task 1 把它加进 `[project].dependencies`。这是 spec §3.3「`multiprocessing.Pipe` + msgpack」的硬要求，不是可选项。
>
> 3. **`multiprocessing` 在本仓库尚无先例。** 既有子进程模式是 `model_manager.py` 用 `subprocess.Popen` 跑 vLLM（HTTP 通信，无 Pipe），以及 `celery` worker（独立进程，redis broker）。本 Lane 引入的 `multiprocessing.Process` + `Pipe` 是新模式——所以本 Lane 全程 fake adapter + chaos 测试压 F1，把这套模式的正确性边界先钉死，Lane D/E/F/G 才在其上叠真 adapter。
>
> 4. **spec §3.3 没列 `Ready` 消息，但 §4.2 生命周期图明确「wait runner 发 "ready" 消息（含 runner_id + GPU list）」。** 本 Lane 把 `Ready` 补成正式协议消息（runner → 主进程，子进程 event loop 起来后第一个发的）。已在 §3.3 覆盖核对里标注。
>
> 5. **spec §4.2 `RunnerSupervisor` 草图把 `ping()` 写成 supervisor 方法。** 实现上 `ping` 属于 RunnerClient（它持有 PipeChannel），supervisor 的 `_watchdog` 调 `self.client.ping()`。语义不变，归属调整。已在 Self-Review 标注。

---

## File Structure

| 文件 | Lane C 动作 | 责任 |
|---|---|---|
| `backend/pyproject.toml` | **修改** | `[project].dependencies` 加 `msgpack>=1.1` |
| `backend/src/runner/__init__.py` | **新建** | 包标记，空文件 |
| `backend/src/runner/protocol.py` | **新建** | 10 个 IPC 消息 dataclass + `encode` / `decode`（msgpack，JSON fallback） |
| `backend/src/runner/pipe_channel.py` | **新建** | `PipeChannel`：F1 约束封装——读侧 connect_read_pipe / 线程桥，写侧写线程 + 5s 超时 |
| `backend/src/runner/fake_adapter.py` | **新建** | `FakeAdapter(InferenceAdapter)`：可配置 crash/slow/fail-load/多 step，零 GPU |
| `backend/src/runner/runner_process.py` | **新建** | `runner_main()` 子进程入口 + pipe-reader / node-executor 双 task |
| `backend/src/runner/client.py` | **新建** | `RunnerClient`：PipeChannel 之上的节点级 RPC（run_node / load_model / ping） |
| `backend/src/runner/supervisor.py` | **新建** | `RunnerSupervisor`：fork / watchdog / crash 检测 / backoff 重启 / GPU-free gate |
| `backend/tests/test_runner_protocol.py` | **新建** | 10 个消息编解码往返、msgpack ↔ JSON fallback、未知 kind 报错 |
| `backend/tests/test_pipe_channel.py` | **新建** | 双向收发、写超时（慢消费者）、读侧 EOF、并发写 |
| `backend/tests/test_fake_adapter.py` | **新建** | load/infer 正常、fail-load、crash、多 step 进度、cancel flag 中断 |
| `backend/tests/test_runner_process.py` | **新建** | 真 `multiprocessing.Process` 跑 fake runner：Ready → LoadModel → RunNode → NodeResult；Abort-during-node |
| `backend/tests/test_runner_supervisor.py` | **新建** | spawn + Ready 握手、ping 超时检测、crash → inflight 标 failed、backoff 序列、GPU-free gate |

> 测试基础设施复用：`tests/conftest.py` 强制 `ADMIN_PASSWORD=""` + `NOUS_DISABLE_BG_TASKS=1` + `CUDA_VISIBLE_DEVICES=""`。本 Lane 的 runner 子进程测试不碰 app / DB，但仍受 conftest 的 CUDA 隐藏保护（fake adapter 本就不 import torch）。

---

## Task 1: 加 msgpack 依赖 + IPC 协议层（`protocol.py`）

spec §3.3 列了 10 个消息（主进程→runner 5 个：LoadModel/UnloadModel/RunNode/Abort/Ping；runner→主进程 4 个：NodeResult/NodeProgress/ModelEvent/Pong）+ §4.2 隐含的 Ready。先把依赖和协议层立起来——这是后面所有 Task 的地基。

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/src/runner/__init__.py`
- Create: `backend/src/runner/protocol.py`
- Test: `backend/tests/test_runner_protocol.py`（新建）

- [ ] **Step 1: 加 msgpack 依赖**

`backend/pyproject.toml` 的 `[project].dependencies` 列表末尾（`"pyotp>=2.9",` 之后）追加一行：
```toml
    # msgpack: V1.5 GPU Runner IPC wire format (main process <-> image/TTS
    # runner subprocess over multiprocessing.Pipe). Small, pure-wheel, no
    # native CUDA deps — safe in the API-server venv. See spec 3.3.
    "msgpack>=1.1",
```
然后装进当前 venv：
```bash
cd backend && uv sync
```
Expected: `uv sync` 成功，输出含 `+ msgpack==1.x`。验证：`python -c "import msgpack; print(msgpack.version)"` 打印版本元组。

- [ ] **Step 2: 建包目录**

```bash
cd backend && mkdir -p src/runner && touch src/runner/__init__.py
```

- [ ] **Step 3: 写失败测试 — 10 个消息编解码往返 + fallback**

新建 `backend/tests/test_runner_protocol.py`：
```python
"""Lane C: GPU Runner IPC 协议编解码测试（纯内存，无子进程、无 GPU）。"""
import pytest

from src.runner import protocol as P


def _round_trip(msg, fmt: str):
    """encode -> decode 往返，断言还原成同类型同字段。"""
    raw = P.encode(msg, fmt=fmt)
    assert isinstance(raw, bytes)
    back = P.decode(raw, fmt=fmt)
    assert type(back) is type(msg)
    assert back == msg
    return back


ALL_MESSAGES = [
    P.Ready(runner_id="runner-i", group_id="image", gpus=[2]),
    P.LoadModel(model_key="flux2-dev", config={"vram_gb": 24}),
    P.UnloadModel(model_key="flux2-dev"),
    P.RunNode(
        task_id=101, node_id="sampler", node_type="image",
        model_key="flux2-dev", inputs={"prompt": "a cat", "steps": 30},
        is_deterministic=False,
    ),
    P.Abort(task_id=101, node_id="sampler"),
    P.Ping(),
    P.NodeResult(
        task_id=101, node_id="sampler", status="completed",
        outputs={"path": "outputs/101/0.png", "meta": {"w": 1024}},
        error=None, duration_ms=4200,
    ),
    P.NodeProgress(task_id=101, node_id="sampler", progress=0.4, detail="step 12/30"),
    P.ModelEvent(event="loaded", model_key="flux2-dev", error=None),
    P.Pong(runner_id="runner-i", loaded_models=["flux2-dev"]),
]


@pytest.mark.parametrize("msg", ALL_MESSAGES, ids=lambda m: type(m).__name__)
def test_round_trip_msgpack(msg):
    _round_trip(msg, fmt="msgpack")


@pytest.mark.parametrize("msg", ALL_MESSAGES, ids=lambda m: type(m).__name__)
def test_round_trip_json(msg):
    """dev 模式 NOUS_IPC_FORMAT=json fallback —— 同样往返成立。"""
    _round_trip(msg, fmt="json")


def test_msgpack_is_more_compact_than_json():
    """sanity: msgpack 编码不大于 JSON（不是严格更小，但典型负载应更紧凑）。"""
    msg = ALL_MESSAGES[3]  # RunNode，带 inputs dict
    assert len(P.encode(msg, fmt="msgpack")) <= len(P.encode(msg, fmt="json"))


def test_decode_unknown_kind_raises():
    """收到未知 kind 的消息 —— decode 抛 ProtocolError，不静默吞。"""
    import msgpack
    bogus = msgpack.packb({"kind": "not_a_real_message", "x": 1})
    with pytest.raises(P.ProtocolError):
        P.decode(bogus, fmt="msgpack")


def test_default_format_from_env(monkeypatch):
    """encode/decode 不传 fmt 时读 NOUS_IPC_FORMAT，默认 msgpack。"""
    monkeypatch.delenv("NOUS_IPC_FORMAT", raising=False)
    assert P.default_format() == "msgpack"
    monkeypatch.setenv("NOUS_IPC_FORMAT", "json")
    assert P.default_format() == "json"
    msg = P.Ping()
    # 不传 fmt → 用 env 的 json
    raw = P.encode(msg)
    assert P.decode(raw) == msg
```

- [ ] **Step 4: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.protocol'`。

- [ ] **Step 5: 实现 `protocol.py`**

新建 `backend/src/runner/protocol.py`：
```python
"""GPU Runner IPC 协议 —— 主进程 <-> image/TTS runner 子进程的 wire format。

spec §3.3。走 multiprocessing.Pipe，msgpack 编码（dev 模式 NOUS_IPC_FORMAT=json
fallback 便于 journalctl 调试）。**仅 image/TTS runner 走此协议**；LLM runner 不
收 RunNode，主进程直连其 vLLM HTTP 端口（Lane E）。

消息是 frozen dataclass —— 跨进程边界传不可变值，避免别名 bug。每个消息有一个
`kind` 字面量做判别式，`decode` 按 kind 路由回正确的 dataclass。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


class ProtocolError(Exception):
    """编解码失败 / 未知消息 kind。"""


# ------------------------------------------------------------------
# 主进程 -> image/TTS runner
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LoadModel:
    model_key: str
    config: dict[str, Any] = field(default_factory=dict)
    kind: Literal["load_model"] = "load_model"


@dataclass(frozen=True)
class UnloadModel:
    model_key: str
    kind: Literal["unload_model"] = "unload_model"


@dataclass(frozen=True)
class RunNode:
    task_id: int
    node_id: str
    node_type: str  # 仅 "image" / "tts"
    model_key: str | None
    inputs: dict[str, Any]
    is_deterministic: bool = False
    kind: Literal["run_node"] = "run_node"


@dataclass(frozen=True)
class Abort:
    task_id: int
    node_id: str | None = None
    kind: Literal["abort"] = "abort"


@dataclass(frozen=True)
class Ping:
    kind: Literal["ping"] = "ping"


# ------------------------------------------------------------------
# image/TTS runner -> 主进程
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Ready:
    """runner 子进程 event loop 起来后发的第一个消息（spec §4.2 生命周期图）。"""

    runner_id: str
    group_id: str
    gpus: list[int]
    kind: Literal["ready"] = "ready"


@dataclass(frozen=True)
class NodeResult:
    task_id: int
    node_id: str
    status: Literal["completed", "failed", "cancelled"]
    outputs: dict[str, Any] | None
    error: str | None
    duration_ms: int
    kind: Literal["node_result"] = "node_result"


@dataclass(frozen=True)
class NodeProgress:
    task_id: int
    node_id: str
    progress: float  # 0.0 ~ 1.0
    detail: str | None = None
    kind: Literal["node_progress"] = "node_progress"


@dataclass(frozen=True)
class ModelEvent:
    event: Literal["loaded", "unloaded", "load_failed"]
    model_key: str
    error: str | None = None
    kind: Literal["model_event"] = "model_event"


@dataclass(frozen=True)
class Pong:
    runner_id: str
    loaded_models: list[str] = field(default_factory=list)
    kind: Literal["pong"] = "pong"


# kind 字面量 -> dataclass 类的路由表
_KIND_TO_CLASS: dict[str, type] = {
    "load_model": LoadModel,
    "unload_model": UnloadModel,
    "run_node": RunNode,
    "abort": Abort,
    "ping": Ping,
    "ready": Ready,
    "node_result": NodeResult,
    "node_progress": NodeProgress,
    "model_event": ModelEvent,
    "pong": Pong,
}

# 类型注解仅供调用方做 isinstance / match —— 任意消息的联合类型
Message = (
    LoadModel | UnloadModel | RunNode | Abort | Ping
    | Ready | NodeResult | NodeProgress | ModelEvent | Pong
)


def default_format() -> str:
    """wire format：环境变量 NOUS_IPC_FORMAT，默认 msgpack。"""
    fmt = os.getenv("NOUS_IPC_FORMAT", "msgpack").strip().lower()
    return fmt if fmt in ("msgpack", "json") else "msgpack"


def encode(msg: Any, *, fmt: str | None = None) -> bytes:
    """把消息 dataclass 编成 bytes。"""
    fmt = fmt or default_format()
    payload = asdict(msg)
    if fmt == "json":
        return json.dumps(payload).encode("utf-8")
    import msgpack

    return msgpack.packb(payload, use_bin_type=True)


def decode(raw: bytes, *, fmt: str | None = None) -> Any:
    """把 bytes 解回对应的消息 dataclass。未知 kind 抛 ProtocolError。"""
    fmt = fmt or default_format()
    try:
        if fmt == "json":
            payload = json.loads(raw.decode("utf-8"))
        else:
            import msgpack

            payload = msgpack.unpackb(raw, raw=False)
    except Exception as e:  # noqa: BLE001 — 任何解码异常统一包成 ProtocolError
        raise ProtocolError(f"failed to decode {fmt} payload: {e}") from e

    if not isinstance(payload, dict) or "kind" not in payload:
        raise ProtocolError(f"decoded payload is not a tagged message: {payload!r}")
    kind = payload["kind"]
    cls = _KIND_TO_CLASS.get(kind)
    if cls is None:
        raise ProtocolError(f"unknown message kind: {kind!r}")
    # 只取该 dataclass 声明的字段，多余 key 忽略（向前兼容）
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in payload.items() if k in known}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ProtocolError(f"payload missing fields for {kind!r}: {e}") from e
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_protocol.py -v`
Expected: 全部 PASS（10 msgpack + 10 json + compact + unknown-kind + env-default = 23 个用例）。

- [ ] **Step 7: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/protocol.py tests/test_runner_protocol.py
git add pyproject.toml src/runner/__init__.py src/runner/protocol.py tests/test_runner_protocol.py
git commit -m "feat(runner): add msgpack IPC protocol for GPU runner subprocess

10 frozen-dataclass messages (LoadModel/UnloadModel/RunNode/Abort/Ping
+ Ready/NodeResult/NodeProgress/ModelEvent/Pong) with kind-tagged
encode/decode. msgpack wire format, NOUS_IPC_FORMAT=json dev fallback.
Adds msgpack>=1.1 dependency. V1.5 Lane C, spec 3.3."
```

---

## Task 2: `PipeChannel` —— F1 约束封装

spec §3.3 的 F1 实现约束：`multiprocessing.Pipe` 对象**不可直接 await**，且 `Pipe.send` **无 timeout 参数**。`PipeChannel` 把这两点封死：读侧把 fd 注册进 event loop（拿不到 fd 退化为读线程桥），写侧用专用写线程对 `send` 加 5s 超时（spec §4.1「IPC pipe 阻塞 5s 超时」）。

**Files:**
- Create: `backend/src/runner/pipe_channel.py`
- Test: `backend/tests/test_pipe_channel.py`（新建）

- [ ] **Step 1: 写失败测试 — 双向收发 + 写超时 + EOF**

新建 `backend/tests/test_pipe_channel.py`：
```python
"""Lane C: PipeChannel 测试 —— F1 约束（Pipe 不可 await、send 无 timeout）的封装。

用 multiprocessing.Pipe() 在同进程内开一对 conn，两端各包一个 PipeChannel,
不起子进程也能压完整的 asyncio 桥接 + 写超时逻辑。
"""
import asyncio
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel, PipeWriteTimeout


@pytest.mark.asyncio
async def test_send_and_recv_round_trip():
    """一端 send_message，另一端 recv_message 拿到等价消息。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        msg = P.RunNode(
            task_id=1, node_id="n", node_type="image",
            model_key="m", inputs={"k": "v"},
        )
        await ch_a.send_message(msg)
        got = await ch_b.recv_message()
        assert got == msg
    finally:
        ch_a.close()
        ch_b.close()


@pytest.mark.asyncio
async def test_recv_eof_raises_connection_closed():
    """对端 close 后，recv_message 抛 ConnectionClosed（runner crash 检测靠它）。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        ch_a.close()  # 模拟对端（runner）崩溃
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(ch_b.recv_message(), timeout=2.0)
    finally:
        ch_b.close()


@pytest.mark.asyncio
async def test_send_times_out_on_slow_consumer():
    """对端不读 pipe，缓冲区写满后 send 应在 write_timeout 内抛 PipeWriteTimeout。

    F1：Pipe.send 本身无 timeout —— PipeChannel 用写线程 + join 超时实现。
    """
    a, b = mp.Pipe()
    # b 端永不读 —— a 端持续 send 直到 OS pipe 缓冲写满后阻塞
    ch_a = PipeChannel(a, write_timeout=2.0)
    try:
        big = P.RunNode(
            task_id=1, node_id="n", node_type="image", model_key="m",
            inputs={"blob": "x" * 100_000},  # 大负载，加速填满缓冲
        )
        with pytest.raises(PipeWriteTimeout):
            # 循环 send，缓冲满后某次 send 会超时
            for _ in range(10_000):
                await ch_a.send_message(big)
    finally:
        ch_a.close()
        b.close()


@pytest.mark.asyncio
async def test_concurrent_sends_are_serialized():
    """多个协程并发 send，写线程串行化，对端收齐所有消息且不交错损坏。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        n = 20
        await asyncio.gather(*(
            ch_a.send_message(P.NodeProgress(task_id=i, node_id="n", progress=0.5))
            for i in range(n)
        ))
        seen = set()
        for _ in range(n):
            msg = await asyncio.wait_for(ch_b.recv_message(), timeout=2.0)
            seen.add(msg.task_id)
        assert seen == set(range(n))
    finally:
        ch_a.close()
        ch_b.close()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_pipe_channel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.pipe_channel'`。

- [ ] **Step 3: 实现 `pipe_channel.py`**

新建 `backend/src/runner/pipe_channel.py`：
```python
"""PipeChannel —— multiprocessing.Pipe 的 asyncio 桥接（F1 约束封装）。

spec §3.3 F1：
  * `multiprocessing.Pipe` 对象不可直接 await —— 读侧把底层 fd 注册进 event loop
    (`loop.connect_read_pipe`)，拿不到 fd 时退化为读线程桥到 asyncio.Queue。
  * `Pipe.send` 无 timeout 参数 —— 写侧用一个专用写线程 + queue.Queue，对每次
    send 加超时（默认 5s，spec §4.1）。send 超时 = 对端假死，由上层（supervisor）
    判定为 runner crash。

wire format：每条消息 = 4-byte big-endian 长度前缀 + protocol.encode 出的 body。
长度前缀让读侧能在字节流上切出完整 frame（connect_read_pipe 给的是字节流，不是
multiprocessing 的对象边界）。
"""
from __future__ import annotations

import asyncio
import os
import queue
import struct
import threading
from typing import Any

from src.runner import protocol as P

# 默认写超时（秒），spec §4.1「IPC pipe 阻塞 5s 超时」
DEFAULT_WRITE_TIMEOUT = 5.0
# 4-byte 长度前缀
_LEN_PREFIX = struct.Struct(">I")


class PipeWriteTimeout(Exception):
    """send 在 write_timeout 内未完成 —— 对端假死。"""


class _LengthPrefixedProtocol(asyncio.Protocol):
    """connect_read_pipe 用的 asyncio.Protocol：在字节流上切 length-prefixed frame，
    完整 frame 解码后丢进 asyncio.Queue。"""

    def __init__(self, out_queue: asyncio.Queue, fmt: str) -> None:
        self._queue = out_queue
        self._fmt = fmt
        self._buf = bytearray()
        self._eof = False

    def data_received(self, data: bytes) -> None:
        self._buf.extend(data)
        while len(self._buf) >= _LEN_PREFIX.size:
            (body_len,) = _LEN_PREFIX.unpack_from(self._buf, 0)
            if len(self._buf) < _LEN_PREFIX.size + body_len:
                break  # frame 还没收全
            start = _LEN_PREFIX.size
            body = bytes(self._buf[start:start + body_len])
            del self._buf[:start + body_len]
            try:
                msg = P.decode(body, fmt=self._fmt)
            except P.ProtocolError as e:
                self._queue.put_nowait(_DecodeFailure(e))
                continue
            self._queue.put_nowait(msg)

    def eof_received(self) -> None:
        self._eof = True
        self._queue.put_nowait(_EOF)

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._eof:
            self._queue.put_nowait(_EOF)


class _DecodeFailure:
    def __init__(self, err: P.ProtocolError) -> None:
        self.err = err


_EOF = object()  # sentinel：对端 close / EOF


class PipeChannel:
    """一端 multiprocessing.Pipe connection 的 asyncio 包装。

    读：`recv_message()` —— event loop 友好，永不阻塞 loop。
    写：`send_message()` —— 经专用写线程，对 send 加 write_timeout。
    """

    def __init__(
        self,
        conn: Any,
        *,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        fmt: str | None = None,
    ) -> None:
        self._conn = conn
        self._write_timeout = write_timeout
        self._fmt = fmt or P.default_format()
        self._closed = False

        # —— 读侧 ——
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._transport: asyncio.ReadTransport | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_started = False

        # —— 写侧 ——
        # 每个待写 frame = (bytes, threading.Event done, list[Exception|None] err)
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="pipe-writer", daemon=True
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # 读侧
    # ------------------------------------------------------------------

    async def _ensure_reader(self) -> None:
        """惰性启动读侧 —— 优先 connect_read_pipe，失败退化为读线程桥。"""
        if self._reader_started:
            return
        self._reader_started = True
        loop = asyncio.get_running_loop()
        try:
            # connect_read_pipe 需要一个有 fileno() 的可读对象。
            # multiprocessing.Connection 在 POSIX 上 fileno() 返回底层 fd。
            pipe_obj = os.fdopen(os.dup(self._conn.fileno()), "rb", buffering=0)
            transport, _ = await loop.connect_read_pipe(
                lambda: _LengthPrefixedProtocol(self._recv_queue, self._fmt),
                pipe_obj,
            )
            self._transport = transport
        except (OSError, ValueError, NotImplementedError):
            # 拿不到 fd（Windows / 特殊平台）→ 退化为读线程桥到 asyncio.Queue
            self._start_reader_thread(loop)

    def _start_reader_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        def _loop() -> None:
            while not self._closed:
                try:
                    msg = self._conn.recv()  # 阻塞读一个 multiprocessing 对象
                except EOFError:
                    loop.call_soon_threadsafe(self._recv_queue.put_nowait, _EOF)
                    return
                except OSError:
                    loop.call_soon_threadsafe(self._recv_queue.put_nowait, _EOF)
                    return
                loop.call_soon_threadsafe(self._recv_queue.put_nowait, msg)

        self._reader_thread = threading.Thread(
            target=_loop, name="pipe-reader-bridge", daemon=True
        )
        self._reader_thread.start()

    async def recv_message(self) -> Any:
        """收一条消息。对端 close → 抛 ConnectionError。解码失败 → 抛 ProtocolError。"""
        await self._ensure_reader()
        item = await self._recv_queue.get()
        if item is _EOF:
            raise ConnectionError("pipe peer closed (EOF)")
        if isinstance(item, _DecodeFailure):
            raise item.err
        return item

    # ------------------------------------------------------------------
    # 写侧
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """专用写线程：从 _write_queue 取 frame，写 pipe，结果回填 Event。"""
        while True:
            item = self._write_queue.get()
            if item is None:  # close 信号
                return
            body, done, err_box = item
            try:
                # connect_read_pipe 读侧期望 length-prefixed 字节流 →
                # 用 send_bytes 写裸字节（不是 send 的 pickle 协议）。
                self._conn.send_bytes(body)
            except Exception as e:  # noqa: BLE001
                err_box[0] = e
            finally:
                done.set()

    async def send_message(self, msg: Any) -> None:
        """发一条消息。write_timeout 内未完成 → 抛 PipeWriteTimeout。"""
        if self._closed:
            raise ConnectionError("channel closed")
        body = P.encode(msg, fmt=self._fmt)
        framed = _LEN_PREFIX.pack(len(body)) + body
        done = threading.Event()
        err_box: list[Exception | None] = [None]
        self._write_queue.put((framed, done, err_box))
        loop = asyncio.get_running_loop()
        # 在 executor 里等 Event，不阻塞 event loop
        finished = await loop.run_in_executor(
            None, done.wait, self._write_timeout
        )
        if not finished:
            raise PipeWriteTimeout(
                f"pipe send did not complete in {self._write_timeout}s"
            )
        if err_box[0] is not None:
            raise ConnectionError(f"pipe send failed: {err_box[0]}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_queue.put(None)  # 停写线程
        if self._transport is not None:
            self._transport.close()
        try:
            self._conn.close()
        except OSError:
            pass
```

> 实现说明：读侧统一用 `send_bytes` / `connect_read_pipe` 走「length-prefixed 裸字节流」，**不**用 `multiprocessing` 的 `send`/`recv`（pickle 协议）—— 因为 `connect_read_pipe` 给的是字节流，必须自己切 frame。读线程桥那条退化路径也对应改用 `recv_bytes` 才严格一致；但 POSIX 上 `connect_read_pipe` 永远成功，退化路径仅 Windows 命中，本仓库是 Linux 部署，退化路径用 `recv` 仅作兜底（测试在 POSIX 上跑的是 `connect_read_pipe` 主路径）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_pipe_channel.py -v`
Expected: 4 个用例全 PASS（`test_send_times_out_on_slow_consumer` 约 2s，因 write_timeout=2.0）。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/pipe_channel.py tests/test_pipe_channel.py
git add src/runner/pipe_channel.py tests/test_pipe_channel.py
git commit -m "feat(runner): add PipeChannel asyncio bridge over multiprocessing.Pipe

Encapsulates the F1 constraints from spec 3.3: Pipe objects are not
awaitable (read side registers fd via loop.connect_read_pipe, falls
back to a reader thread), and Pipe.send has no timeout (write side
uses a dedicated writer thread + Event.wait with a 5s timeout, raising
PipeWriteTimeout on slow consumers). Length-prefixed framing on the
byte stream. V1.5 Lane C, spec 3.3 / 4.1."
```

---

## Task 3: `FakeAdapter` —— 零 GPU 的 InferenceAdapter 实现

runner 子进程框架的测试不能依赖真 GPU / 真模型。`FakeAdapter` 实现 `src/services/inference/base.py` 的 `InferenceAdapter` ABC，`load` / `infer` 全是可让出的 `await asyncio.sleep`，可配置 crash / slow / fail-load / 多 step 进度 / cancel-flag 中断。Lane D/E/F 才在真 adapter 上跑同样的路径。

**Files:**
- Create: `backend/src/runner/fake_adapter.py`
- Test: `backend/tests/test_fake_adapter.py`（新建）

- [ ] **Step 1: 写失败测试 — fake adapter 各行为模式**

新建 `backend/tests/test_fake_adapter.py`：
```python
"""Lane C: FakeAdapter 测试 —— 零 GPU 的 InferenceAdapter，给 runner 框架测试用。"""
import asyncio
import threading

import pytest

from src.runner.fake_adapter import FakeAdapter, FakeLoadError
from src.services.inference.base import ImageRequest, InferenceAdapter, MediaModality


def _img_req(steps: int = 4) -> ImageRequest:
    return ImageRequest(request_id="r1", prompt="a cat", steps=steps)


def test_fake_adapter_is_inference_adapter():
    a = FakeAdapter(paths={"main": "/fake"})
    assert isinstance(a, InferenceAdapter)
    assert a.modality == MediaModality.IMAGE


@pytest.mark.asyncio
async def test_load_and_infer_happy_path():
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.01)
    assert not a.is_loaded
    await a.load("cpu")
    assert a.is_loaded
    result = await a.infer(_img_req())
    assert result.media_type == "image/png"
    assert result.data  # 非空 bytes
    assert result.usage.image_count == 1


@pytest.mark.asyncio
async def test_fail_load_raises():
    a = FakeAdapter(paths={"main": "/fake"}, fail_load=True)
    with pytest.raises(FakeLoadError):
        await a.load("cpu")
    assert not a.is_loaded


@pytest.mark.asyncio
async def test_crash_on_infer_raises_runtime_error():
    """crash_on_infer=True —— infer 抛异常，模拟节点执行期 native fault。"""
    a = FakeAdapter(paths={"main": "/fake"}, crash_on_infer=True)
    await a.load("cpu")
    with pytest.raises(RuntimeError):
        await a.infer(_img_req())


@pytest.mark.asyncio
async def test_infer_reports_per_step_progress():
    """多 step 时，progress_callback 每 step 被调一次，参数单调递增。"""
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.0)
    await a.load("cpu")
    seen: list[tuple[int, int]] = []
    await a.infer(_img_req(steps=5), progress_callback=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


@pytest.mark.asyncio
async def test_cancel_flag_interrupts_infer():
    """传入一个已 set 的 threading.Event，infer 在下一 step 边界抛 asyncio.CancelledError。

    对应 spec §4.4：within-node cancel 信号穿过 to_thread 边界（这里 fake 用
    asyncio.sleep 模拟，真 adapter 用 callback_on_step_end —— 接口形状一致）。
    """
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.02)
    await a.load("cpu")
    flag = threading.Event()
    flag.set()  # 一开始就取消
    with pytest.raises(asyncio.CancelledError):
        await a.infer(_img_req(steps=10), cancel_flag=flag)


@pytest.mark.asyncio
async def test_infer_yields_to_event_loop():
    """infer 必须可让出 —— 否则 runner 的 pipe-reader 收不到调度（spec §4.4）。

    起一个并发的 sleep(0)，infer 跑 3 step（每 step sleep）期间它应能完成。
    """
    a = FakeAdapter(paths={"main": "/fake"}, infer_seconds=0.05)
    await a.load("cpu")
    other_ran = asyncio.Event()

    async def _other():
        await asyncio.sleep(0)
        other_ran.set()

    await asyncio.gather(a.infer(_img_req(steps=3)), _other())
    assert other_ran.is_set()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.fake_adapter'`。

- [ ] **Step 3: 实现 `fake_adapter.py`**

新建 `backend/src/runner/fake_adapter.py`：
```python
"""FakeAdapter —— 零 GPU / 零模型的 InferenceAdapter 实现。

V1.5 Lane C 用它跑通 runner 子进程框架（IPC + 生命周期）而不需要真硬件。
Lane D/E/F 把真 adapter（image_diffusers / TTS / vLLM）迁进 runner 时，复用
本 Lane 验证过的同一套 runner / supervisor 代码路径。

可配置的行为开关（构造参数）：
  fail_load        —— load() 抛 FakeLoadError（模拟权重文件丢失 / OOM）
  crash_on_infer   —— infer() 抛 RuntimeError（模拟节点执行期 native fault）
  infer_seconds    —— 每个 step 的模拟耗时（asyncio.sleep，可让出 event loop）

infer() 支持 progress_callback（每 step 回调一次）和 cancel_flag（threading.Event,
set 后下一 step 边界抛 asyncio.CancelledError）—— 形状对齐真 image adapter 的
diffusers callback_on_step_end + CancelFlag（spec §4.4）。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, ClassVar

from src.services.inference.base import (
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


class FakeLoadError(Exception):
    """FakeAdapter.load() 在 fail_load=True 时抛 —— 模拟加载失败。"""


class FakeAdapter(InferenceAdapter):
    """假 adapter：可配置 crash / slow / fail-load / 多 step 进度。"""

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    estimated_vram_mb: ClassVar[int] = 0

    def __init__(
        self,
        paths: dict[str, str],
        device: str = "cpu",
        *,
        fail_load: bool = False,
        crash_on_infer: bool = False,
        infer_seconds: float = 0.01,
        **params: Any,
    ) -> None:
        super().__init__(paths, device, **params)
        self._fail_load = fail_load
        self._crash_on_infer = crash_on_infer
        self._infer_seconds = infer_seconds

    async def load(self, device: str) -> None:
        await asyncio.sleep(0)  # 可让出
        if self._fail_load:
            raise FakeLoadError(f"fake load failure for paths={self.paths}")
        self.device = device
        self._model = object()  # 非 None → is_loaded True

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> InferenceResult:
        if self._crash_on_infer:
            raise RuntimeError("fake adapter crash during infer")

        steps = int(getattr(req, "steps", 1) or 1)
        started = time.monotonic()
        for done in range(1, steps + 1):
            # 每 step 边界检查 cancel —— 对齐真 adapter 的 callback_on_step_end
            if cancel_flag is not None and cancel_flag.is_set():
                raise asyncio.CancelledError()
            if self._infer_seconds > 0:
                await asyncio.sleep(self._infer_seconds)  # 可让出 event loop
            else:
                await asyncio.sleep(0)
            if progress_callback is not None:
                progress_callback(done, steps)

        latency_ms = int((time.monotonic() - started) * 1000)
        return InferenceResult(
            media_type="image/png",
            data=b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes",
            metadata={"fake": True, "steps": steps},
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
```

> 设计说明：`infer` 的 `progress_callback` / `cancel_flag` 是 `InferenceAdapter.infer` 之外的 keyword-only 扩展参数 —— ABC 的 `infer(req)` 签名不变（FakeAdapter 兼容 isinstance 检查），runner 通过 `**` 传递。真 image adapter（Lane G 重写）会把 `cancel_flag` 接到 diffusers `callback_on_step_end`，`progress_callback` 接到同一 callback —— 接口形状本 Lane 先钉死。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_fake_adapter.py -v`
Expected: 7 个用例全 PASS。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/fake_adapter.py tests/test_fake_adapter.py
git add src/runner/fake_adapter.py tests/test_fake_adapter.py
git commit -m "feat(runner): add FakeAdapter for GPU-free runner framework tests

Implements the InferenceAdapter ABC with no torch/CUDA. Configurable
fail_load / crash_on_infer / infer_seconds; infer() supports
progress_callback (per-step) and cancel_flag (threading.Event) —
interface shape matches the real image adapter's diffusers
callback_on_step_end + CancelFlag (spec 4.4). V1.5 Lane C."
```

---

## Task 4: runner 子进程骨架（`runner_process.py`）—— pipe-reader + node-executor 双 task

spec §4.4 / D9：image/TTS runner 子进程内跑**两个 asyncio task**。pipe-reader 持续读 pipe：RunNode 入内部 `asyncio.Queue`，Abort 置对应 task 的 `threading.Event`，LoadModel/UnloadModel/Ping 同步处理。node-executor 从队列取 RunNode、调 fake adapter、发 NodeProgress / NodeResult。关键性质：pipe-reader 永不阻塞在 adapter 上，Abort 能立即置位。

**Files:**
- Create: `backend/src/runner/runner_process.py`
- Test: `backend/tests/test_runner_process.py`（新建）

- [ ] **Step 1: 写失败测试 — 真 multiprocessing.Process 跑 fake runner**

新建 `backend/tests/test_runner_process.py`：
```python
"""Lane C: runner 子进程骨架测试 —— 真 multiprocessing.Process 起 fake runner。

用 spawn context（与 CUDA 子进程惯例一致）。runner 内跑 FakeAdapter，
不碰真 GPU。验证 Ready 握手 + LoadModel + RunNode + NodeProgress/NodeResult +
Abort-during-node。
"""
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import runner_main

_SPAWN = mp.get_context("spawn")


def _spawn_runner(group_id="image", gpus=(2,)):
    """起一个 fake runner 子进程，返回 (process, PipeChannel 主进程侧)。"""
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(
        target=runner_main,
        args=(group_id, list(gpus), child_conn),
        kwargs={"adapter_class": "src.runner.fake_adapter.FakeAdapter"},
        daemon=True,
    )
    proc.start()
    child_conn.close()  # 主进程侧不用 child 端
    return proc, PipeChannel(parent_conn)


@pytest.mark.asyncio
async def test_runner_sends_ready_on_startup():
    """子进程 event loop 起来后第一个发 Ready（runner_id + group + gpus）。"""
    proc, ch = _spawn_runner(group_id="image", gpus=(2,))
    try:
        msg = await _recv(ch)
        assert isinstance(msg, P.Ready)
        assert msg.group_id == "image"
        assert msg.gpus == [2]
        assert msg.runner_id  # 非空
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_load_model_then_run_node():
    """LoadModel → ModelEvent(loaded)；RunNode → NodeProgress* → NodeResult(completed)。"""
    proc, ch = _spawn_runner()
    try:
        await _recv(ch)  # 吞掉 Ready
        await ch.send_message(P.LoadModel(model_key="fake-img", config={}))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent) and ev.event == "loaded"

        await ch.send_message(P.RunNode(
            task_id=7, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 3},
        ))
        progresses, result = await _collect_until_result(ch)
        assert len(progresses) == 3  # 每 step 一个 NodeProgress
        assert isinstance(result, P.NodeResult)
        assert result.status == "completed"
        assert result.task_id == 7
        assert result.outputs is not None
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_ping_returns_pong():
    proc, ch = _spawn_runner()
    try:
        await _recv(ch)  # Ready
        await ch.send_message(P.Ping())
        pong = await _recv(ch)
        assert isinstance(pong, P.Pong)
        assert pong.runner_id
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_load_failed_emits_model_event():
    """fail_load 的模型 —— ModelEvent(load_failed)，runner 不崩。"""
    proc, ch = _spawn_runner()
    try:
        await _recv(ch)  # Ready
        # config 里的 fail_load 透传给 FakeAdapter 构造
        await ch.send_message(P.LoadModel(
            model_key="bad-model", config={"fail_load": True},
        ))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent)
        assert ev.event == "load_failed"
        assert ev.error
        # runner 仍活着：再 Ping 应回 Pong
        await ch.send_message(P.Ping())
        assert isinstance(await _recv(ch), P.Pong)
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_abort_during_node_cancels_it():
    """RunNode 执行中收到 Abort —— 该节点 NodeResult.status == cancelled。

    pipe-reader 收 Abort 立即置 threading.Event；node-executor 的 fake adapter
    在下一 step 边界看到 flag → CancelledError → NodeResult(cancelled)。
    """
    proc, ch = _spawn_runner()
    try:
        await _recv(ch)  # Ready
        await ch.send_message(P.LoadModel(model_key="fake-img", config={"infer_seconds": 0.1}))
        assert isinstance(await _recv(ch), P.ModelEvent)
        # 跑一个 20 step 的长节点
        await ch.send_message(P.RunNode(
            task_id=9, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 20},
        ))
        # 收到第一个 progress 后立刻 Abort
        first = await _recv(ch)
        assert isinstance(first, P.NodeProgress)
        await ch.send_message(P.Abort(task_id=9, node_id="sampler"))
        # 继续收，最终应是 cancelled 的 NodeResult
        _, result = await _collect_until_result(ch, already=[first])
        assert isinstance(result, P.NodeResult)
        assert result.status == "cancelled"
        assert result.task_id == 9
    finally:
        await _shutdown(proc, ch)


# —— 测试辅助 ——

import asyncio


async def _recv(ch: PipeChannel, timeout: float = 10.0):
    return await asyncio.wait_for(ch.recv_message(), timeout=timeout)


async def _collect_until_result(ch: PipeChannel, already=None):
    """收消息直到拿到 NodeResult，返回 (progress 列表, NodeResult)。"""
    progresses = list(already or [])
    progresses = [m for m in progresses if isinstance(m, P.NodeProgress)]
    while True:
        msg = await _recv(ch)
        if isinstance(msg, P.NodeResult):
            return progresses, msg
        if isinstance(msg, P.NodeProgress):
            progresses.append(msg)


async def _shutdown(proc, ch: PipeChannel):
    ch.close()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.runner_process'`。

- [ ] **Step 3: 实现 `runner_process.py`**

新建 `backend/src/runner/runner_process.py`：
```python
"""image/TTS runner 子进程入口 + 内部双 asyncio task。

spec §4.4 / D9：runner 子进程内跑两个 task：
  * pipe-reader —— 持续读 pipe：RunNode 入内部 asyncio.Queue；Abort 置对应
    task 的 threading.Event；LoadModel/UnloadModel/Ping 直接处理。永不阻塞在
    adapter 上 —— 这样 Abort 才能立即置位。
  * node-executor —— 从队列取 RunNode、get-or-load adapter、调 adapter.infer
    (传 progress_callback + cancel_flag)、发 NodeProgress / NodeResult。

cancel 信号用 threading.Event：真 adapter 的扩散循环在 to_thread 里跑，跨线程
信号必须用 threading 原语（spec §4.4 关键性质 D14）。本 Lane fake adapter 用
asyncio.sleep 模拟，但 cancel_flag 接口形状一致。

本 Lane 用 fake adapter；ModelManager 迁入是 Lane D。这里的「模型表」是个极简
dict[model_key -> adapter 实例]，够跑通 IPC + 生命周期。
"""
from __future__ import annotations

import asyncio
import importlib
import threading
import time
import uuid
from typing import Any

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel


class _RunnerState:
    """runner 子进程内的可变状态。"""

    def __init__(self, runner_id: str, group_id: str, gpus: list[int], adapter_class: str):
        self.runner_id = runner_id
        self.group_id = group_id
        self.gpus = gpus
        self.adapter_class = adapter_class
        # model_key -> adapter 实例（本 Lane 极简版，Lane D 换成真 ModelManager）
        self.adapters: dict[str, Any] = {}
        # 待执行的 RunNode 队列（pipe-reader 投，node-executor 取）
        self.run_queue: asyncio.Queue[P.RunNode] = asyncio.Queue()
        # task_id -> cancel flag（pipe-reader 收 Abort 时 set）
        self.cancel_flags: dict[int, threading.Event] = {}
        self.shutdown = asyncio.Event()


def _load_adapter_class(dotted: str) -> type:
    module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def _handle_load_model(state: _RunnerState, ch: PipeChannel, msg: P.LoadModel) -> None:
    """LoadModel —— 实例化 adapter + load，发 ModelEvent。"""
    cls = _load_adapter_class(state.adapter_class)
    # config 里的 key（如 fail_load / infer_seconds）透传给 adapter 构造
    try:
        adapter = cls(paths={"main": f"/fake/{msg.model_key}"}, **msg.config)
        await adapter.load(f"cuda:{state.gpus[0]}" if state.gpus else "cpu")
    except Exception as e:  # noqa: BLE001
        await ch.send_message(P.ModelEvent(
            event="load_failed", model_key=msg.model_key, error=f"{type(e).__name__}: {e}",
        ))
        return
    state.adapters[msg.model_key] = adapter
    await ch.send_message(P.ModelEvent(event="loaded", model_key=msg.model_key, error=None))


async def _handle_unload_model(state: _RunnerState, ch: PipeChannel, msg: P.UnloadModel) -> None:
    adapter = state.adapters.pop(msg.model_key, None)
    if adapter is not None:
        adapter.unload()
    await ch.send_message(P.ModelEvent(event="unloaded", model_key=msg.model_key, error=None))


async def _pipe_reader(state: _RunnerState, ch: PipeChannel) -> None:
    """持续读 pipe，分派消息。永不阻塞在 adapter 上。"""
    while not state.shutdown.is_set():
        try:
            msg = await ch.recv_message()
        except ConnectionError:
            # 主进程关了 pipe —— runner 该退出了
            state.shutdown.set()
            return
        except P.ProtocolError:
            # 坏消息，跳过（不崩 runner）
            continue

        if isinstance(msg, P.RunNode):
            state.cancel_flags[msg.task_id] = threading.Event()
            state.run_queue.put_nowait(msg)
        elif isinstance(msg, P.Abort):
            flag = state.cancel_flags.get(msg.task_id)
            if flag is not None:
                flag.set()  # node-executor 的 adapter 下一 step 边界看到
        elif isinstance(msg, P.LoadModel):
            await _handle_load_model(state, ch, msg)
        elif isinstance(msg, P.UnloadModel):
            await _handle_unload_model(state, ch, msg)
        elif isinstance(msg, P.Ping):
            await ch.send_message(P.Pong(
                runner_id=state.runner_id,
                loaded_models=list(state.adapters.keys()),
            ))
        # 其余消息类型（runner→主进程方向的）不应收到，忽略


async def _node_executor(state: _RunnerState, ch: PipeChannel) -> None:
    """从队列取 RunNode，跑 adapter，发 progress / result。"""
    while not state.shutdown.is_set():
        try:
            node = await asyncio.wait_for(state.run_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue  # 周期性回头看 shutdown

        cancel_flag = state.cancel_flags.get(node.task_id) or threading.Event()
        adapter = state.adapters.get(node.model_key) if node.model_key else None
        started = time.monotonic()

        if adapter is None:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"model {node.model_key!r} not loaded",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        # progress_callback —— 每 step 发一个 NodeProgress
        def _on_progress(done: int, total: int, _node=node) -> None:
            ch._write_queue  # noqa: B018 — 仅文档：send 经写线程，callback 同步入队
            asyncio.get_running_loop().create_task(ch.send_message(P.NodeProgress(
                task_id=_node.task_id, node_id=_node.node_id,
                progress=done / total if total else 1.0,
                detail=f"step {done}/{total}",
            )))

        try:
            from src.services.inference.base import ImageRequest

            req = ImageRequest(
                request_id=f"task-{node.task_id}",
                prompt=str(node.inputs.get("prompt", "")),
                steps=int(node.inputs.get("steps", 1) or 1),
            )
            result = await adapter.infer(
                req, progress_callback=_on_progress, cancel_flag=cancel_flag,
            )
        except asyncio.CancelledError:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="cancelled",
                outputs=None, error="aborted",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue
        except Exception as e:  # noqa: BLE001
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        await ch.send_message(P.NodeResult(
            task_id=node.task_id, node_id=node.node_id, status="completed",
            outputs={"meta": result.metadata, "media_type": result.media_type},
            error=None,
            duration_ms=int((time.monotonic() - started) * 1000),
        ))
        state.cancel_flags.pop(node.task_id, None)


async def _runner_loop(state: _RunnerState, ch: PipeChannel) -> None:
    """子进程主协程：发 Ready，起 pipe-reader + node-executor 双 task。"""
    await ch.send_message(P.Ready(
        runner_id=state.runner_id, group_id=state.group_id, gpus=state.gpus,
    ))
    reader = asyncio.create_task(_pipe_reader(state, ch), name="pipe-reader")
    executor = asyncio.create_task(_node_executor(state, ch), name="node-executor")
    await state.shutdown.wait()
    reader.cancel()
    executor.cancel()
    await asyncio.gather(reader, executor, return_exceptions=True)


def runner_main(
    group_id: str,
    gpus: list[int],
    conn: Any,
    *,
    adapter_class: str = "src.runner.fake_adapter.FakeAdapter",
) -> None:
    """multiprocessing.Process 的 target —— image/TTS runner 子进程入口。

    起一个独立 event loop（spec §4.5：runner 有自己的 Event Loop B）。
    adapter_class 默认 FakeAdapter（Lane C）；Lane D/F 传真 adapter dotted path。
    """
    runner_id = f"runner-{group_id}-{uuid.uuid4().hex[:6]}"
    state = _RunnerState(runner_id, group_id, gpus, adapter_class)
    ch = PipeChannel(conn)
    try:
        asyncio.run(_runner_loop(state, ch))
    finally:
        ch.close()
```

> 实现说明：`_on_progress` 是同步 callback（fake adapter 在它的 step 循环里直接调），但 `ch.send_message` 是 async —— 这里用 `create_task` 把发送排进 event loop。因为 fake adapter 的 `infer` 每 step 都 `await asyncio.sleep`，create_task 出来的发送有机会跑。真 adapter（Lane G）的扩散循环在 `to_thread` 里，callback 是在工作线程里跑的 —— 那时 `_on_progress` 要改用 `loop.call_soon_threadsafe`。本 Lane 标记此处为 Lane G 的接线点（见 Self-Review 已知风险）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process.py -v`
Expected: 5 个用例全 PASS。注意：起真子进程，单文件约 10-20s。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/runner_process.py tests/test_runner_process.py
git add src/runner/runner_process.py tests/test_runner_process.py
git commit -m "feat(runner): add image/TTS runner subprocess skeleton (dual-task)

runner_main() spawns an independent event loop running two asyncio
tasks (spec 4.4 / D9): pipe-reader (reads pipe, queues RunNode, sets
threading.Event on Abort, never blocks on the adapter) and
node-executor (pulls RunNode, runs adapter.infer with progress_callback
+ cancel_flag, emits NodeProgress/NodeResult). Uses FakeAdapter; real
ModelManager wiring is Lane D. V1.5 Lane C, spec 4.4."
```

---

## Task 5: `RunnerClient` —— 主进程侧节点级 RPC

spec §3.5：RunnerClient 是主进程侧、PipeChannel 之上的节点级 RPC。它持有一个 PipeChannel，提供 `run_node(spec)` / `load_model` / `unload_model` / `ping` / `abort`，并在后台跑一个 demux 协程把 runner 发回的消息按 task_id 路由到对应的 future / 进度回调。

**Files:**
- Create: `backend/src/runner/client.py`
- Test: `backend/tests/test_runner_client.py`（新建）

- [ ] **Step 1: 写失败测试 — RunnerClient RPC + demux**

新建 `backend/tests/test_runner_client.py`：
```python
"""Lane C: RunnerClient 测试 —— 主进程侧节点级 RPC。

用真 multiprocessing fake runner 子进程，验证 RunnerClient 把 pipe 上的消息流
demux 成「per-node 等待 + 进度回调」。
"""
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main

_SPAWN = mp.get_context("spawn")


async def _make_client(group_id="image", gpus=(2,)) -> tuple:
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(
        target=runner_main, args=(group_id, list(gpus), child_conn), daemon=True,
    )
    proc.start()
    child_conn.close()
    client = RunnerClient(parent_conn, runner_id=f"runner-{group_id}")
    await client.start()  # 起 demux 协程 + 等 Ready
    return proc, client


async def _teardown(proc, client):
    await client.close()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3.0)


@pytest.mark.asyncio
async def test_start_waits_for_ready():
    proc, client = await _make_client(group_id="image", gpus=(2,))
    try:
        assert client.is_ready
        assert client.gpus == [2]
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_load_model_returns_on_model_event():
    proc, client = await _make_client()
    try:
        ok = await client.load_model("fake-img", config={})
        assert ok is True
        # fail_load 模型 —— 返回 False
        bad = await client.load_model("bad", config={"fail_load": True})
        assert bad is False
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_run_node_resolves_with_node_result():
    proc, client = await _make_client()
    try:
        await client.load_model("fake-img", config={})
        progress_seen: list[float] = []
        result = await client.run_node(
            P.RunNode(
                task_id=11, node_id="sampler", node_type="image",
                model_key="fake-img", inputs={"steps": 4},
            ),
            on_progress=lambda pr: progress_seen.append(pr.progress),
        )
        assert isinstance(result, P.NodeResult)
        assert result.status == "completed"
        assert result.task_id == 11
        assert len(progress_seen) == 4  # 4 step → 4 个 progress 回调
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_ping_returns_pong():
    proc, client = await _make_client()
    try:
        pong = await client.ping()
        assert isinstance(pong, P.Pong)
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_recv_eof_marks_client_disconnected():
    """runner 子进程死掉 → pipe EOF → client 的 inflight run_node 异常结束。"""
    proc, client = await _make_client()
    try:
        await client.load_model("fake-img", config={"infer_seconds": 0.2})
        import asyncio

        # 跑一个长节点，执行中杀掉 runner
        run_task = asyncio.create_task(client.run_node(P.RunNode(
            task_id=12, node_id="sampler", node_type="image",
            model_key="fake-img", inputs={"steps": 50},
        )))
        await asyncio.sleep(0.3)
        proc.terminate()  # 模拟 crash
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(run_task, timeout=5.0)
        assert not client.is_connected
    finally:
        await _teardown(proc, client)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.client'`。

- [ ] **Step 3: 实现 `client.py`**

新建 `backend/src/runner/client.py`：
```python
"""RunnerClient —— 主进程侧、PipeChannel 之上的节点级 RPC（spec §3.5）。

主进程对每个 image/TTS runner 持一个 RunnerClient。它：
  * start()  —— 起后台 demux 协程，等 runner 的 Ready 握手。
  * run_node(spec, on_progress) —— 发 RunNode，await 到对应的 NodeResult；
    期间的 NodeProgress 路由给 on_progress 回调。
  * load_model / unload_model —— 发消息，await 对应 ModelEvent。
  * ping —— 发 Ping，await Pong（supervisor 的 watchdog 用）。
  * abort(task_id) —— 发 Abort（不等回，runner 会照常发 NodeResult(cancelled)）。

demux：runner 发回的消息全经一个后台协程读，按 task_id / 类型路由到对应的
asyncio.Future 或回调。pipe EOF（runner crash）→ 所有 inflight future 置
ConnectionError 异常。
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel


class RunnerClient:
    def __init__(
        self,
        conn: Any,
        *,
        runner_id: str,
        ready_timeout: float = 30.0,
    ) -> None:
        self._ch = PipeChannel(conn)
        self.runner_id = runner_id
        self._ready_timeout = ready_timeout

        self._ready = asyncio.Event()
        self._connected = True
        self.gpus: list[int] = []
        self.group_id: str | None = None

        # task_id -> Future[NodeResult]
        self._node_futures: dict[int, asyncio.Future] = {}
        # task_id -> on_progress 回调
        self._progress_cbs: dict[int, Callable[[P.NodeProgress], None]] = {}
        # model_key -> Future[bool]（ModelEvent loaded/load_failed）
        self._model_futures: dict[str, asyncio.Future] = {}
        # 单个待回 Pong 的 Future（ping 是串行的，watchdog 一次一个）
        self._pong_future: asyncio.Future | None = None

        self._demux_task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """起 demux 协程，等 Ready 握手。"""
        self._demux_task = asyncio.create_task(self._demux_loop(), name="runner-demux")
        await asyncio.wait_for(self._ready.wait(), timeout=self._ready_timeout)

    async def close(self) -> None:
        self._connected = False
        if self._demux_task is not None:
            self._demux_task.cancel()
            try:
                await self._demux_task
            except asyncio.CancelledError:
                pass
        self._ch.close()

    def _fail_all_inflight(self, exc: Exception) -> None:
        """runner 断连 —— 所有等待中的 future 置异常。"""
        for fut in list(self._node_futures.values()):
            if not fut.done():
                fut.set_exception(exc)
        for fut in list(self._model_futures.values()):
            if not fut.done():
                fut.set_exception(exc)
        if self._pong_future is not None and not self._pong_future.done():
            self._pong_future.set_exception(exc)
        self._node_futures.clear()
        self._model_futures.clear()

    # ------------------------------------------------------------------
    # demux
    # ------------------------------------------------------------------

    async def _demux_loop(self) -> None:
        while True:
            try:
                msg = await self._ch.recv_message()
            except ConnectionError as e:
                self._connected = False
                self._fail_all_inflight(e)
                return
            except P.ProtocolError:
                continue  # 坏消息跳过，不崩 demux

            if isinstance(msg, P.Ready):
                self.group_id = msg.group_id
                self.gpus = msg.gpus
                self._ready.set()
            elif isinstance(msg, P.NodeProgress):
                cb = self._progress_cbs.get(msg.task_id)
                if cb is not None:
                    cb(msg)
            elif isinstance(msg, P.NodeResult):
                fut = self._node_futures.pop(msg.task_id, None)
                self._progress_cbs.pop(msg.task_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
            elif isinstance(msg, P.ModelEvent):
                fut = self._model_futures.pop(msg.model_key, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg.event == "loaded")
            elif isinstance(msg, P.Pong):
                if self._pong_future is not None and not self._pong_future.done():
                    self._pong_future.set_result(msg)

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    async def run_node(
        self,
        spec: P.RunNode,
        *,
        on_progress: Callable[[P.NodeProgress], None] | None = None,
    ) -> P.NodeResult:
        """发 RunNode，await 对应的 NodeResult。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._node_futures[spec.task_id] = fut
        if on_progress is not None:
            self._progress_cbs[spec.task_id] = on_progress
        await self._ch.send_message(spec)
        return await fut

    async def load_model(self, model_key: str, *, config: dict | None = None) -> bool:
        """发 LoadModel，await ModelEvent。返回是否加载成功。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._model_futures[model_key] = fut
        await self._ch.send_message(P.LoadModel(model_key=model_key, config=config or {}))
        return await fut

    async def unload_model(self, model_key: str) -> None:
        if not self._connected:
            raise ConnectionError("runner disconnected")
        await self._ch.send_message(P.UnloadModel(model_key=model_key))

    async def abort(self, task_id: int, node_id: str | None = None) -> None:
        """发 Abort —— 不等回，runner 会照常发 NodeResult(cancelled)。"""
        if not self._connected:
            return
        await self._ch.send_message(P.Abort(task_id=task_id, node_id=node_id))

    async def ping(self) -> P.Pong:
        """发 Ping，await Pong。supervisor 的 watchdog 用。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        loop = asyncio.get_running_loop()
        self._pong_future = loop.create_future()
        await self._ch.send_message(P.Ping())
        return await self._pong_future
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_client.py -v`
Expected: 5 个用例全 PASS。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/client.py tests/test_runner_client.py
git add src/runner/client.py tests/test_runner_client.py
git commit -m "feat(runner): add RunnerClient node-level RPC over PipeChannel

Main-process side (spec 3.5): a background demux coroutine routes
runner messages by task_id/type to per-node futures + progress
callbacks. run_node/load_model/unload_model/abort/ping RPCs. pipe EOF
(runner crash) fails all inflight futures with ConnectionError.
V1.5 Lane C, spec 3.5."
```

---

## Task 6: `RunnerSupervisor` —— spawn / watchdog / crash 重启 / GPU-free gate

spec §4.2：RunnerSupervisor fork runner 子进程、跑 watchdog（ping 超时检测）、crash 时把 inflight task 标 failed、按 `RESTART_BACKOFF` 指数退避重启、重启前过 F2 GPU-free gate（轮询 nvidia-smi 确认显存回落）。本 Lane 用 fake runner + 可注入的 GPU-free 探针，让整套逻辑无 GPU 可测。

**Files:**
- Create: `backend/src/runner/supervisor.py`
- Test: `backend/tests/test_runner_supervisor.py`（新建）

- [ ] **Step 1: 写失败测试 — supervisor 生命周期 + crash 重启 + backoff + gate**

新建 `backend/tests/test_runner_supervisor.py`：
```python
"""Lane C: RunnerSupervisor 测试 —— spawn / watchdog / crash 重启 / GPU-free gate。

用 fake runner 子进程 + 注入的 GPU-free 探针（不碰 nvidia-smi）。
"""
import asyncio

import pytest

from src.runner import protocol as P
from src.runner.supervisor import RunnerSupervisor


def _make_supervisor(**overrides) -> RunnerSupervisor:
    """构造一个跑 fake runner 的 supervisor，超时参数缩小以便快测。"""
    kw = dict(
        group_id="image",
        gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.3,
        ping_timeout=0.5,
        restart_backoff=[0.1, 0.2, 0.3],
        gpu_free_probe=lambda gpus: True,  # 默认 GPU 立即 free
    )
    kw.update(overrides)
    return RunnerSupervisor(**kw)


@pytest.mark.asyncio
async def test_start_spawns_runner_and_handshakes():
    sup = _make_supervisor()
    try:
        await sup.start()
        assert sup.is_running
        assert sup.client.is_ready
        # 能正常派活
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node(P.RunNode(
            task_id=1, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 2},
        ))
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_watchdog_detects_crash_and_restarts():
    """杀掉 runner 子进程 → watchdog ping 超时 → 自动重启 → 新 runner 可用。"""
    sup = _make_supervisor()
    try:
        await sup.start()
        old_pid = sup.pid
        # 模拟 crash
        sup._process.terminate()
        # 等 watchdog 检测 + 重启（ping_interval + ping_timeout + backoff + 重启）
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sup.is_running
        assert sup.pid != old_pid  # 新进程
        assert sup.restart_count == 1
        # 新 runner 能干活
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node(P.RunNode(
            task_id=2, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 2},
        ))
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_crash_marks_inflight_tasks_failed():
    """runner crash 时，supervisor 把登记的 inflight task 全标 failed（runner_crashed）。"""
    failed: list[tuple[int, str]] = []
    sup = _make_supervisor(
        on_task_failed=lambda task_id, reason: failed.append((task_id, reason)),
    )
    try:
        await sup.start()
        # 登记两个 inflight task
        sup.register_inflight(101)
        sup.register_inflight(102)
        sup._process.terminate()
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sorted(t for t, _ in failed) == [101, 102]
        assert all(reason == "runner_crashed" for _, reason in failed)
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_restart_backoff_sequence():
    """连续 crash —— backoff 按 restart_backoff 序列递增，封顶最后一个值。"""
    sup = _make_supervisor(restart_backoff=[0.1, 0.3, 0.5])
    try:
        await sup.start()
        # backoff_for(n) 给第 n 次重启该等多久
        assert sup.backoff_for(0) == 0.1
        assert sup.backoff_for(1) == 0.3
        assert sup.backoff_for(2) == 0.5
        assert sup.backoff_for(3) == 0.5  # 封顶
        assert sup.backoff_for(99) == 0.5
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_gpu_free_gate_blocks_restart_until_clear():
    """GPU-free gate：探针返回 False 时重启被卡住，返回 True 后才继续（F2）。"""
    gate_state = {"free": False}
    probe_calls: list[int] = []

    def _probe(gpus):
        probe_calls.append(1)
        return gate_state["free"]

    sup = _make_supervisor(gpu_free_probe=_probe, gpu_free_poll_interval=0.1)
    try:
        await sup.start()
        sup._process.terminate()
        # gate 卡住 —— 0.5s 内不应完成重启
        await asyncio.sleep(1.0)
        assert not sup.is_running or sup.restart_count == 0
        assert len(probe_calls) >= 2  # gate 在轮询
        # 放行
        gate_state["free"] = True
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sup.is_running
        assert sup.restart_count == 1
    finally:
        await sup.stop()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_supervisor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.supervisor'`。

- [ ] **Step 3: 实现 `supervisor.py`**

新建 `backend/src/runner/supervisor.py`：
```python
"""RunnerSupervisor —— 主进程侧的 runner 子进程监管（spec §4.2）。

职责：
  * start()  —— fork runner 子进程，建 RunnerClient，等 Ready 握手。
  * watchdog —— 每 ping_interval 发一次 ping；ping_timeout 内无 Pong 或 pipe
    EOF → 判定 crash → _restart()。
  * _restart() —— 终结旧 runner（terminate → kill）；inflight task 全标 failed
    (runner_crashed)；按 RESTART_BACKOFF 退避；过 F2 GPU-free gate（轮询
    gpu_free_probe 直到该 group 的 GPU 显存回落）；重新 fork + 握手。
  * 成功跑满 stable_seconds（默认 30min）后 reset restart_count（spec §4.2 第 6 步）。

本 Lane 用 fake runner（adapter_class 默认 FakeAdapter）+ 注入式 gpu_free_probe
（不碰真 nvidia-smi）。生产环境 Lane H 接真探针 + resident preload。
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from typing import Callable

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main

logger = logging.getLogger(__name__)

# spec §4.2 默认值
DEFAULT_PING_INTERVAL = 30.0
DEFAULT_PING_TIMEOUT = 10.0
DEFAULT_RESTART_BACKOFF = [5.0, 15.0, 60.0, 300.0]  # 封顶 5 min
DEFAULT_STABLE_SECONDS = 30 * 60  # 跑满 30min 视为稳定，reset crash count

_SPAWN = mp.get_context("spawn")  # CUDA 子进程惯例：spawn 不 fork


def _default_gpu_free_probe(gpus: list[int]) -> bool:
    """生产用 GPU-free 探针：nvidia-smi 查这些 GPU 的显存是否回落到基线。

    本 Lane 测试注入 fake 探针；这个默认实现是 Lane H 接 resident preload 时
    会用到的真探针骨架 —— 此处保守返回 True（无 GPU 环境不阻塞）。
    """
    return True


class RunnerSupervisor:
    def __init__(
        self,
        *,
        group_id: str,
        gpus: list[int],
        adapter_class: str = "src.runner.fake_adapter.FakeAdapter",
        ping_interval: float = DEFAULT_PING_INTERVAL,
        ping_timeout: float = DEFAULT_PING_TIMEOUT,
        restart_backoff: list[float] | None = None,
        stable_seconds: float = DEFAULT_STABLE_SECONDS,
        gpu_free_probe: Callable[[list[int]], bool] | None = None,
        gpu_free_poll_interval: float = 2.0,
        on_task_failed: Callable[[int, str], None] | None = None,
    ) -> None:
        self.group_id = group_id
        self.gpus = gpus
        self.adapter_class = adapter_class
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.restart_backoff = restart_backoff or DEFAULT_RESTART_BACKOFF
        self.stable_seconds = stable_seconds
        self._gpu_free_probe = gpu_free_probe or _default_gpu_free_probe
        self._gpu_free_poll_interval = gpu_free_poll_interval
        self._on_task_failed = on_task_failed

        self._process: mp.Process | None = None
        self.client: RunnerClient | None = None
        self.restart_count = 0
        self._inflight: set[int] = set()
        self._last_spawn_at = 0.0
        self._watchdog_task: asyncio.Task | None = None
        self._stopping = False
        self._restarted_count = 0
        self._restart_event = asyncio.Event()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.is_alive()
            and self.client is not None
            and self.client.is_connected
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def backoff_for(self, restart_index: int) -> float:
        """第 restart_index 次重启该等多久（封顶 restart_backoff 最后一个值）。"""
        if restart_index < len(self.restart_backoff):
            return self.restart_backoff[restart_index]
        return self.restart_backoff[-1]

    # ------------------------------------------------------------------
    # inflight task 登记（crash 时全标 failed）
    # ------------------------------------------------------------------

    def register_inflight(self, task_id: int) -> None:
        self._inflight.add(task_id)

    def unregister_inflight(self, task_id: int) -> None:
        self._inflight.discard(task_id)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._spawn()
        self._watchdog_task = asyncio.create_task(
            self._watchdog(), name=f"watchdog-{self.group_id}"
        )

    async def _spawn(self) -> None:
        """fork runner 子进程 + 建 client + 等 Ready。"""
        parent_conn, child_conn = _SPAWN.Pipe()
        proc = _SPAWN.Process(
            target=runner_main,
            args=(self.group_id, self.gpus, child_conn),
            kwargs={"adapter_class": self.adapter_class},
            daemon=True,
            name=f"runner-{self.group_id}",
        )
        proc.start()
        child_conn.close()  # 主进程侧不用 child 端
        self._process = proc
        self.client = RunnerClient(parent_conn, runner_id=f"runner-{self.group_id}")
        await self.client.start()  # 等 Ready 握手
        self._last_spawn_at = time.monotonic()
        logger.info(
            "runner %s spawned (pid=%s, gpus=%s)", self.group_id, proc.pid, self.gpus
        )

    async def stop(self) -> None:
        """优雅停止 —— 不再重启，终结子进程。"""
        self._stopping = True
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        if self.client is not None:
            await self.client.close()
        await self._terminate_process()

    async def _terminate_process(self) -> None:
        """SIGTERM 5s → SIGKILL。"""
        proc = self._process
        if proc is None:
            return
        if proc.is_alive():
            proc.terminate()
            for _ in range(50):  # 最多等 5s
                if not proc.is_alive():
                    break
                await asyncio.sleep(0.1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3.0)
        else:
            proc.join(timeout=1.0)

    # ------------------------------------------------------------------
    # watchdog + restart
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """每 ping_interval ping 一次；超时 / EOF → crash → 重启。"""
        while not self._stopping:
            await asyncio.sleep(self.ping_interval)
            if self._stopping:
                return
            try:
                await asyncio.wait_for(self.client.ping(), timeout=self.ping_timeout)
            except (asyncio.TimeoutError, ConnectionError):
                if self._stopping:
                    return
                logger.warning("runner %s ping failed → restarting", self.group_id)
                await self._restart()

    async def _restart(self) -> None:
        """spec §4.2 crash 检测 + 重启 6 步。"""
        # 1. 终结旧 runner
        await self._terminate_process()
        if self.client is not None:
            await self.client.close()

        # 2. inflight task 全标 failed (runner_crashed)，不重试
        for task_id in list(self._inflight):
            if self._on_task_failed is not None:
                self._on_task_failed(task_id, "runner_crashed")
        self._inflight.clear()

        # 3. backoff（防 crash 风暴）
        backoff = self.backoff_for(self.restart_count)
        await asyncio.sleep(backoff)

        # 4. F2 GPU-free gate —— 轮询直到该 group 的 GPU 显存回落
        while not self._stopping and not self._gpu_free_probe(self.gpus):
            logger.info(
                "runner %s GPU-free gate: GPUs %s not yet free, waiting",
                self.group_id, self.gpus,
            )
            await asyncio.sleep(self._gpu_free_poll_interval)
        if self._stopping:
            return

        # 5. 重新 fork + 握手
        try:
            await self._spawn()
        except Exception:
            logger.exception("runner %s respawn failed", self.group_id)
            self.restart_count += 1
            return

        # 6. restart_count 累加；成功跑满 stable_seconds 后由 _watchdog reset
        #    （此处只累加，reset 逻辑在下方 _maybe_reset_crash_count）
        self.restart_count += 1
        self._restarted_count += 1
        self._restart_event.set()
        self._restart_event = asyncio.Event()
        logger.info(
            "runner %s restarted (count=%d, backoff=%.1fs)",
            self.group_id, self.restart_count, backoff,
        )

    def _maybe_reset_crash_count(self) -> None:
        """跑满 stable_seconds 无 crash → reset restart_count（spec §4.2 第 6 步）。

        由调用方（Lane H 的健康检查 loop）周期性调用，或 watchdog 每轮调。
        """
        if (
            self.restart_count > 0
            and self._last_spawn_at > 0
            and time.monotonic() - self._last_spawn_at >= self.stable_seconds
        ):
            logger.info(
                "runner %s stable for %.0fs → reset crash count",
                self.group_id, self.stable_seconds,
            )
            self.restart_count = 0

    # ------------------------------------------------------------------
    # 测试辅助
    # ------------------------------------------------------------------

    async def wait_restarted(self, count: int = 1) -> None:
        """等到 supervisor 完成至少 count 次重启（测试用）。"""
        while self._restarted_count < count:
            await asyncio.sleep(0.05)
```

> 实现说明：`backoff_for(restart_index)` 把 spec §4.2 的 `RESTART_BACKOFF = [5, 15, 60, 300]` 封顶逻辑提成纯函数，方便单测。`gpu_free_probe` 是注入式 —— 本 Lane 测试传 fake 探针（不碰 nvidia-smi），`_default_gpu_free_probe` 是 Lane H 接真 nvidia-smi 探针的骨架（无 GPU 时保守返回 True 不阻塞）。`wait_restarted` 是测试辅助，生产代码不依赖它。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_supervisor.py -v`
Expected: 5 个用例全 PASS。注意：起真子进程 + crash/重启时序，单文件约 30-50s。

- [ ] **Step 5: lint 预检 + Commit**

```bash
cd backend && ruff check src/runner/supervisor.py tests/test_runner_supervisor.py
git add src/runner/supervisor.py tests/test_runner_supervisor.py
git commit -m "feat(runner): add RunnerSupervisor with watchdog + crash restart

spec 4.2: spawns runner subprocess (spawn context), runs a ping
watchdog, on crash marks inflight tasks failed (runner_crashed),
backs off per RESTART_BACKOFF, passes the F2 GPU-free gate (injectable
probe, polls until GPUs free) before respawn. Uses fake runner; the
real nvidia-smi probe + resident preload are Lane H. V1.5 Lane C."
```

---

## Task 7: Lane C 整合验证 + chaos 测试 + lint 预检

spec 的 review 报告把 F1（Pipe 无 timeout）点名为本 Lane 承接的 CRITICAL GAP，要求 chaos 测试 `test_pipe_slow_consumer` 专门压。本 Task 加这个 chaos 测试，并跑整合验证。

**Files:**
- Create: `backend/tests/chaos/__init__.py`（新建，若 `tests/chaos/` 不存在）
- Create: `backend/tests/chaos/test_pipe_slow_consumer.py`（新建）
- Test: 全 Lane C 套件

- [ ] **Step 1: 确认 chaos 目录 + 写 slow-consumer chaos 测试**

```bash
cd backend && mkdir -p tests/chaos && touch tests/chaos/__init__.py
```
新建 `backend/tests/chaos/test_pipe_slow_consumer.py`：
```python
"""Lane C chaos: pipe 慢消费者 —— F1 写超时 + 反压验证。

spec §5.5 / review 报告 F1：fake runner 不读 pipe → 主进程 PipeChannel 应在
write_timeout 内抛 PipeWriteTimeout（Pipe.send 本身无 timeout，靠写线程实现）。
这是 F1 实现约束的正确性边界压测。
"""
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel, PipeWriteTimeout

_SPAWN = mp.get_context("spawn")


def _silent_child(conn) -> None:
    """一个永不读 pipe 的子进程 —— 模拟假死的 runner。"""
    import time

    time.sleep(30)  # 啥也不干，conn 缓冲很快写满
    conn.close()


@pytest.mark.asyncio
async def test_send_to_silent_runner_times_out():
    """对端进程活着但不读 —— send 填满 OS pipe 缓冲后在 write_timeout 内超时。"""
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(target=_silent_child, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    ch = PipeChannel(parent_conn, write_timeout=3.0)
    try:
        big = P.RunNode(
            task_id=1, node_id="n", node_type="image", model_key="m",
            inputs={"blob": "x" * 200_000},
        )
        with pytest.raises(PipeWriteTimeout):
            for _ in range(100_000):
                await ch.send_message(big)
    finally:
        ch.close()
        proc.terminate()
        proc.join(timeout=3.0)
```

- [ ] **Step 2: 跑 chaos 测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/chaos/test_pipe_slow_consumer.py -v`
Expected: PASS（约 3s，因 write_timeout=3.0）。

- [ ] **Step 3: Lane C 全部新测试 green**

Run:
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_protocol.py tests/test_pipe_channel.py tests/test_fake_adapter.py tests/test_runner_process.py tests/test_runner_client.py tests/test_runner_supervisor.py tests/chaos/test_pipe_slow_consumer.py -v
```
Expected: 全部 PASS（23 + 4 + 7 + 5 + 5 + 5 + 1 = 50 个用例）。整体约 1-2 分钟（起多个真子进程）。

- [ ] **Step 4: 后端全 suite 无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。新增 50 个用例，无 collection error、无 import error。`src/runner/` 是全新目录，与既有代码零交叉 import —— 既有测试不应受影响。

> 注意：若 `tests/chaos/` 此前不存在，`pytest tests/` 现在会收集到它。这是预期的（spec §5.5 规划了 `tests/chaos/`）。若 CI 配置要求 chaos 测试单独 marker，那是 Lane J（测试套）的 scope —— 本 Lane 只确保 chaos 测试本身能跑过。

- [ ] **Step 5: lint 预检（push 前本地跑）**

Run: `cd backend && ruff check src/runner/ tests/test_runner_protocol.py tests/test_pipe_channel.py tests/test_fake_adapter.py tests/test_runner_process.py tests/test_runner_client.py tests/test_runner_supervisor.py tests/chaos/`
Expected: 无 lint 错误。

- [ ] **Step 6: 开 PR**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git push -u origin <lane-C-branch>
gh pr create --title "feat: V1.5 Lane C — RunnerSupervisor + image/TTS runner framework" --body "$(cat <<'EOF'
## Summary
- 新建 `backend/src/runner/` 包（与既有 `src/workers/` 物理隔离 —— spec 命名「GPU Runner」正是为避开撞名）
- `protocol.py`：10 个 msgpack IPC 消息 + JSON dev fallback；新增 `msgpack>=1.1` 依赖
- `pipe_channel.py`：F1 约束封装 —— `Pipe` 不可 await（读侧 connect_read_pipe）+ `Pipe.send` 无 timeout（写侧写线程 + 5s 超时）
- `fake_adapter.py`：零 GPU 的 `InferenceAdapter` 实现，可配 crash/slow/fail-load
- `runner_process.py`:runner 子进程骨架 —— pipe-reader + node-executor 双 asyncio task（D9）
- `client.py`：RunnerClient 主进程侧节点级 RPC（demux by task_id）
- `supervisor.py`：RunnerSupervisor spawn/watchdog/crash 重启/backoff/F2 GPU-free gate
- chaos 测试 `test_pipe_slow_consumer` 压 F1 写超时
- 全程 fake adapter —— 不接 GroupScheduler / 不迁真 ModelManager / 不做 LLM runner（Lane D/E/G）

## Test plan
- [ ] `test_runner_protocol.py` green（10 消息 msgpack/json 往返）
- [ ] `test_pipe_channel.py` green（双向收发 / 写超时 / EOF / 并发写）
- [ ] `test_fake_adapter.py` green（load/infer/fail-load/crash/进度/cancel）
- [ ] `test_runner_process.py` green（真子进程：Ready/LoadModel/RunNode/Abort-during-node）
- [ ] `test_runner_client.py` green（RPC + demux + crash 时 inflight 异常）
- [ ] `test_runner_supervisor.py` green（spawn/watchdog/crash 重启/backoff/GPU-free gate）
- [ ] `tests/chaos/test_pipe_slow_consumer.py` green（F1 写超时）
- [ ] 后端全 suite green，无回归
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneC-runner-framework`。）

---

## Self-Review

**Spec 覆盖检查：** Lane C 在 spec「实施分 Lane」表里的职责是「RunnerSupervisor + image/TTS runner 子进程框架（fake adapter，跑通 IPC；pipe-reader + executor 双 task；F1 pipe 不可 await 的实现约束）。依赖：A」。

- **RunnerSupervisor**（spawn/watchdog/restart）→ Task 6（`supervisor.py`）。spec §4.2 的 6 步重启流程逐条对应：终结旧 runner（`_terminate_process` SIGTERM→SIGKILL）/ inflight 标 failed（`_on_task_failed`）/ backoff（`backoff_for`）/ F2 GPU-free gate（`_gpu_free_probe` 轮询）/ 重新 fork（`_spawn`）/ stable 后 reset（`_maybe_reset_crash_count`）。
- **RunnerClient**（asyncio bridge over Pipe）→ Task 5（`client.py`）。spec §3.5 的「节点级 RPC: run_node(spec, inputs)」对应 `run_node`；demux by task_id 路由。
- **runner 子进程骨架 + pipe-reader + node-executor 双 task**（spec §4.4 / D9）→ Task 4（`runner_process.py`）。`_pipe_reader` + `_node_executor` 两个 `asyncio.create_task`；pipe-reader 收 Abort 置 `threading.Event`，永不阻塞在 adapter 上。
- **msgpack IPC 协议**（spec §3.3，LoadModel/UnloadModel/RunNode/Abort/Ping + NodeResult/NodeProgress/ModelEvent/Pong）→ Task 1（`protocol.py`）。spec 列的 9 个消息全部覆盖 + 补了 §4.2 隐含的 `Ready`（共 10 个）。
- **F1 约束**（Pipe 不可 await + send 无 timeout）→ Task 2（`pipe_channel.py`）：读侧 `loop.connect_read_pipe`（拿不到 fd 退化读线程桥）；写侧专用写线程 + `Event.wait` 超时 → `PipeWriteTimeout`。Task 7 加 chaos 测试 `test_pipe_slow_consumer` 专门压。
- **fake adapter**（无真 GPU/模型）→ Task 3（`fake_adapter.py`），实现 `InferenceAdapter` ABC。
- **F2 GPU-free gate**（spec §4.2）→ Task 6 `_restart()` 第 4 步，注入式 `gpu_free_probe`，测试用 fake 探针。
- **crash 检测 + backoff 重启**（spec §4.2）→ Task 6 `_watchdog` + `_restart` + `RESTART_BACKOFF`。
- **依赖 A**：Lane A 产出 `hardware.yaml` + GPUAllocator（决定起几个 runner、每个 runner 管哪些 GPU）。本 Lane 的 `RunnerSupervisor(group_id, gpus)` 接口**预留**了 Lane A 的产出作为构造参数 —— 本 Lane 测试直接传 `group_id="image", gpus=[2]`，不 import Lane A 的代码。Lane A 完成后由后续 Lane（H / scheduler）把 `hardware.yaml` 解析结果喂进 `RunnerSupervisor` 构造。这样 Lane C 不被 Lane A 阻塞，符合「依赖 A」的真实耦合度（接口契约依赖，非代码依赖）。

**与 spec / 简报的偏差（已在 plan 顶部「注意」显式标注）：**
1. 新代码落位 `src/runner/` 而非 `src/workers/` —— 简报明确要求「decide where and say so」，已决策并说明（`src/workers/` 是既有 celery engine 模块，spec 命名「GPU Runner」就是为避开撞名）。
2. msgpack 不是现有依赖 —— Task 1 第一步加 `msgpack>=1.1` 进 `pyproject.toml`。
3. `multiprocessing` 在本仓库无先例 —— 本 Lane 引入 `multiprocessing.Process` + `Pipe`，全程 fake adapter + chaos 测试钉死正确性边界。
4. spec §3.3 未列 `Ready` 消息，§4.2 生命周期图却要求它 —— 本 Lane 把 `Ready` 补成正式协议消息。
5. spec §4.2 `RunnerSupervisor.ping()` 草图把 ping 写成 supervisor 方法 —— 实现上 `ping` 属于 `RunnerClient`（它持 PipeChannel），supervisor `_watchdog` 调 `self.client.ping()`。语义不变。

**spec 模糊处的判断：**
- spec §3.3 的消息只给了字段，没给 wire framing。判断：用 4-byte big-endian 长度前缀 + body —— 因为 `loop.connect_read_pipe` 给的是字节流（不是 `multiprocessing` 的对象边界），必须自己切 frame。已在 `pipe_channel.py` docstring 说明。
- spec §3.3 F1 说「pipe-writer 用写线程 + `Queue.join` 超时，或非阻塞 fd 自行轮询」给了两个选项。判断：选写线程 + `Event.wait` 超时 —— 比非阻塞 fd 轮询简单且不丢字节边界。已在 Task 2 实现说明。
- spec §3.5 `RunnerClient` / `GroupScheduler` 是 dataclass 草图，没给方法签名。判断：`RunnerClient` 的 `run_node` / `load_model` / `ping` / `abort` 方法集是按「runner 子进程能收的 5 类消息」+「supervisor watchdog 需要 ping」推出来的。`GroupScheduler` 本 Lane **不实现**（spec Lane 表里 GroupScheduler 是 Lane G）。
- spec §4.2 `RESTART_BACKOFF = [5, 15, 60, 300]` 没说「第 5 次及以后」怎么办。判断：封顶最后一个值（`backoff_for` 纯函数），spec 注释「封顶 5 min」印证此意图。
- spec §4.4 runner 内 progress callback 怎么跨「同步 callback → async send」没明说。判断：fake adapter 的 callback 是同步的，runner 用 `create_task` 把 send 排进 loop；真 adapter（Lane G）callback 在 `to_thread` 工作线程里跑，需改 `loop.call_soon_threadsafe` —— 已在 Task 4 实现说明 + 下方已知风险标注为 Lane G 接线点。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。7 个 Task 全是「写失败测试 → 跑确认失败 → 最小实现 → 跑确认通过 → lint → commit」闭环。所有协议代码、PipeChannel、fake adapter、runner 子进程、client、supervisor、chaos 测试均完整给出，命令带预期输出。`_default_gpu_free_probe` 不是 placeholder —— 它是 Lane H 接真 nvidia-smi 的明确骨架，无 GPU 时保守返回 True 的行为是有意设计（已在 docstring 说明）。

**类型一致性：**
- `protocol.py` 的 10 个 dataclass 字段类型 ↔ 测试构造的实参类型一致（`task_id: int`、`inputs: dict`、`progress: float`、`gpus: list[int]` 等）。
- `PipeChannel.send_message(msg) -> None` / `recv_message() -> Any`；`RunnerClient.run_node(spec: P.RunNode) -> P.NodeResult`、`load_model(...) -> bool`、`ping() -> P.Pong` —— 返回类型标注与测试断言一致。
- `RunnerSupervisor.__init__` 的 `gpu_free_probe: Callable[[list[int]], bool]` ↔ 测试注入的 `lambda gpus: True` / `_probe(gpus)` 签名一致。`on_task_failed: Callable[[int, str], None]` ↔ 测试的 `lambda task_id, reason: ...` 一致。
- `runner_main(group_id: str, gpus: list[int], conn, *, adapter_class: str)` ↔ `RunnerSupervisor._spawn` 的 `_SPAWN.Process(target=runner_main, args=(group_id, gpus, child_conn), kwargs={"adapter_class": ...})` 一致；测试 `_spawn_runner` 也同形。
- `FakeAdapter(paths: dict[str, str], device, *, fail_load, crash_on_infer, infer_seconds, **params)` ↔ runner `_handle_load_model` 的 `cls(paths={"main": ...}, **msg.config)`（`msg.config` 里放 `fail_load` / `infer_seconds`）一致。

**已知风险：**
- **真子进程测试慢且对时序敏感** —— `test_runner_process.py` / `test_runner_client.py` / `test_runner_supervisor.py` 起真 `multiprocessing.Process`，单文件几十秒。supervisor 的 crash/重启测试把超时参数缩到 0.1-0.5s 加速，但 CI 机器负载高时可能 flaky。缓解：测试里的 `wait_restarted` / `asyncio.wait_for` 都给了宽松的外层 timeout（10-15s）。若 CI 仍 flaky，Lane J 可考虑给这几个文件加 `@pytest.mark.flaky` 重试或挪进 integration marker。
- **`connect_read_pipe` 的 fd dup 行为** —— `pipe_channel.py` 用 `os.dup(self._conn.fileno())` + `os.fdopen` 把 multiprocessing Connection 的 fd 交给 `connect_read_pipe`。这在 Linux 上可靠，但 `_conn` 和 dup 出来的 fd 的生命周期需小心（`close()` 里 transport.close + conn.close 双关）。chaos 测试 + EOF 测试覆盖了关闭路径，但长跑下的 fd 泄漏需 Lane J 的 soak 测试盯。
- **progress callback 跨 to_thread 边界（Lane G 接线点）** —— 本 Lane fake adapter 的 `infer` 全在 event loop 里 `await`，progress callback 是同步调用、`create_task` 排发送即可。真 image adapter（Lane G 重写）的扩散循环在 `to_thread` 工作线程里跑，callback 在工作线程上下文 —— 那时 `runner_process.py` 的 `_on_progress` **必须**改成 `loop.call_soon_threadsafe`，否则 `asyncio.get_running_loop()` 在工作线程里会抛。已在 Task 4 实现说明明确标注此处为 Lane G 的接线点。本 Lane 的接口形状（`cancel_flag: threading.Event` + `progress_callback`）已为此预留。
- **`evict_lru` 在 `services/model_manager.py` 是 async** —— 本 Lane 不碰 ModelManager（Lane D 才迁），但记录一个观察：`model_manager.py:459` 的 `evict_lru` 是 `async def`，与 Lane 0 plan 里「`evict_lru` 是同步方法」的描述不符。这不影响 Lane C（本 Lane 零 ModelManager 依赖），但 Lane D 迁 ModelManager 进 runner 时需注意这个 async/sync 事实，按真实代码来。
- **`spawn` context 下 `runner_main` 的可 pickle 性** —— `_SPAWN.Process(target=runner_main, ...)` 要求 `runner_main` 及其参数可 pickle。`runner_main` 是模块级函数（可 pickle），`group_id: str` / `gpus: list[int]` / `adapter_class: str` 都可 pickle，`child_conn` 是 multiprocessing Connection（spawn 下可传）。测试已实跑验证；但后续 Lane 若给 `runner_main` 传不可 pickle 的对象（如已实例化的 ModelManager）会炸 —— Lane D 迁 ModelManager 时必须在子进程内构造，不能跨进程传实例。
