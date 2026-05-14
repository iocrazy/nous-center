# Workflow Queue + GPU Scheduler (V1.5) Design

**Author:** heygo
**Date:** 2026-05-13
**Status:** Draft (pending implementation plan)

## Problem

V1 起 image / TTS / LLM 全在 API server 主进程内串行执行，存在三类问题需要**同时**解决：

1. **并发竞态** — `image_diffusers.generate` 等节点没有 inference-level lock，并发请求会撞同一 `adapter.pipe`，输出错乱甚至 CUDA OOM 拖死整个进程。
2. **GPU 分配粗放** — 当前 model_manager 只看 `gpu_index`，不感知 NVLink 拓扑；vLLM tensor-parallel 模型有可能落到非 NVLink 的卡对上，PCIe 通信成瓶颈。
3. **多模型 resident 缺策略** — 启动时按 yaml 顺序一个一个 load，某个 OOM 阻断后面的，API server 起不来。

V1.5 把这三件事放在**一个架构**下解决：进程隔离 + 拓扑感知调度 + 鲁棒 preload。

### 硬件背景

| 卡 | VRAM | NVLink | 角色 |
|---|---|---|---|
| GPU 0 (3090) | 24 GB | 与 GPU 1 配对 | LLM tensor-parallel |
| GPU 1 (3090) | 24 GB | 与 GPU 0 配对 | LLM tensor-parallel |
| GPU 2 (Pro 6000 Blackwell) | 96 GB | 单卡 | image |
| (可选 GPU 3) | 24 GB | 单卡 | TTS |

CPU AMD Threadripper 7965WX 24C/48T，93 GB RAM。

### 不在 V1.5 范围

- 节点输出缓存（按输入 hash skip 已算过的节点）—— 见 [V1.6 取舍](#v15-vs-v16-节点输出缓存取舍)
- 跨机分布式（永远不做，本机推理 infra）
- 用户 quota / rate limit（单 admin，不需要）
- 模型自动下载/管理（已有独立路径，不重做）

## Goals

1. 同模型/同 GPU 任务**串行**执行，不同 GPU group 任务**并发**执行
2. NVLink-aware 自动分组；vLLM tp 模型只能落 NVLink:true group
3. Worker crash 不拖死主进程；resident preload OOM 不阻断 API server 启动
4. ExecutionTask 历史可观测：queued/started/finished 时间、worker_id、gpu_group、node_timings
5. 前端 TaskPanel + Dashboard 暴露 per-group 队列与 worker 状态
6. Cancel 双层语义：节点边界 + within-node（image sampler 每 step、LLM streaming 每 token）

## Non-Goals

- 节点输出缓存（V1.6）
- 跨机调度
- 优先级超过 2 级（interactive / batch 够用）

---

## 1. 整体架构

### 1.1 关键决策汇总

| 决策点 | V1.5 方案 |
|---|---|
| 进程模型 | 主进程 asyncio 调度 + 每 GPU group 一个 worker 子进程 |
| Worker 拓扑 | 3 worker（image / llm-tp / tts），按 NVLink 物理拓扑划分 |
| GPU 分组 | Runtime `nvidia-smi topo -m` 探测 + `hardware.yaml` override |
| 任务粒度 | Workflow 入队为单 task；executor 内部按节点 GPU 需求分组并发 |
| 优先级 | 2 级（interactive=0 > batch=10），同级 FIFO |
| Cancel | 节点边界 + within-node（sampler step / LLM token） |
| 跨 GPU 数据传输 | CUDA IPC handle（同 group P2P，跨 group 退化 host pinned） |
| Inference lock | per-model `asyncio.Lock` 在 worker 内，wrap 整个 `adapter.run()` |
| History | `execution_tasks` 扩 schema + 主进程 ring buffer（200 entries） |
| Resident 启动 | `preload_order` 数字升序，失败 fail-soft，暴露到 `/health` |
| 协议层 | multiprocessing.Pipe + msgpack（dev 模式 JSON fallback） |
| Frontend UX | 扩 TaskPanel（per-group 折叠区） + Dashboard GpuPanel（worker 标识） |

### 1.2 模块拓扑

```
┌────────────────────────────────────────────────────────────────────┐
│  Main Process (API server, asyncio loop)                           │
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ FastAPI     │  │ Scheduler    │  │ WorkflowExecutor         │  │
│  │ /v1/...     │─▶│ - per-group  │─▶│ - 解析节点 DAG           │  │
│  │             │  │   PriorityQ  │  │ - 按 GPU group 分组      │  │
│  │             │  │ - ring buf   │  │ - 组内串行 / 组间并发    │  │
│  └─────────────┘  └──────────────┘  └──────────────────────────┘  │
│         │                │                     │                   │
│         │                ▼                     ▼                   │
│         │       ┌─────────────────────────────────────────────┐   │
│         │       │ WorkerClient (主进程侧)                     │   │
│         │       │  - asyncio.Queue ↔ multiprocessing pipe    │   │
│         │       │  - 节点级 RPC: run_node(spec, inputs)       │   │
│         │       └─────────────────────────────────────────────┘   │
│         │              │              │              │             │
│         ▼              ▼              ▼              ▼             │
│  ┌──────────┐   ┌──────────┐  ┌──────────────┐  ┌──────────┐     │
│  │ Postgres │   │ Worker-A │  │ Worker-B     │  │ Worker-C │     │
│  │ tasks +  │   │ (image)  │  │ (llm-tp)     │  │ (tts)    │     │
│  │ schema   │   │ Pro 6000 │  │ GPU 0+1 NVL  │  │ GPU 2/3  │     │
│  └──────────┘   ├──────────┤  ├──────────────┤  ├──────────┤     │
│                 │ per-model│  │ vLLM proc    │  │ per-model│     │
│                 │ lock     │  │ (already     │  │ lock     │     │
│                 │ Flux2/SD │  │  subprocess) │  │ Cosy2/...│     │
│                 └──────────┘  └──────────────┘  └──────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 三件事如何统一解决

- **并发竞态** → Scheduler 的 per-group PriorityQueue 解决"同 GPU 任务串行"；worker 内的 per-model lock 解决"同模型 adapter race"。两层 lock 各司其职。
- **NVLink-aware 调度** → 启动 topo 探测 + yaml override 产出 `groups[]`；allocator 按 group 分配；tp 模型在 spec 校验层就强制 nvlink:true。
- **Resident preload** → ModelManager 启动钩子按 `preload_order` 升序 dispatch 到对应 worker；fail-soft 写 `_load_failures`，暴露到 `/health`。

---

## 2. 数据流

### 2.1 完整路径（workflow 提交 → 完成）

```
[API/UI] ──POST /v1/workflows/{id}/run──▶ [Scheduler]
  1. 解析 workflow → 收集模型依赖
  2. 计算所需 GPU groups
  3. 创建 ExecutionTask (status=queued, queued_at=now)
  4. 推入对应 group 的 PriorityQueue
                                              ▼
                                         [Group Dispatcher]
  5. 弹出 (priority, queued_at, task_id) 最小者
  6. 标记 status=running, started_at=now
                                              ▼
                                         [WorkflowExecutor]
  7. DAG 拓扑排序
  8. 按节点 GPU group 分组：
     - 同 group 节点串行
     - 不同 group 节点 asyncio.gather() 并发
                                              ▼
                                         [WorkerClient.run_node(spec)]
  9. 序列化输入（tensor 走 IPCRef）→ multiprocessing pipe → worker
                                              ▼
                                         [Worker 子进程]
  10. ModelManager.get_or_load(model_key)（per-model lock）
  11. adapter.run(inputs, cancel_token)（inference lock 持有）
  12. 返回 outputs
                                              ▼
                                         [WorkflowExecutor]
  13. nodes_done++, current_node 更新
  14. WebSocket 推 progress 事件
                                              ▼
                                         (所有节点完成)
  15. status=completed, finished_at, duration_ms
  16. 写 result, node_timings JSON
  17. ring buffer push
  18. invalidate cache: ["tasks"]
```

### 2.2 时序与一致性

**入队阶段（主进程）**
- ExecutionTask 在 DB commit 之后才入 PriorityQueue —— 保证宕机重启可恢复（启动时扫 status=queued 的 task 重新入队）
- 入队 sort key = `(priority, queued_at, task_id)` —— 同 priority 内 queued_at FIFO

**节点分组阶段**
- 节点 spec `device` 字段 → 解析到 GPU group id
- 同 group 节点队列串行 await，不同 group `asyncio.gather()` 并发
- 跨节点 tensor 输出：同 worker → 直接 dict 传递；跨 worker → CUDA IPC handle

**Worker 内部**
- ModelManager 每 worker 独立实例（不再是模块全局单例）
- per-model `asyncio.Lock` wrap 整个 `adapter.run()`
- LRU evict 只在本 worker 管的 GPU group 范围内做

**Cancel 路径**
- `POST /v1/tasks/{id}/cancel` → 主进程 scheduler 设置 task.cancel_event
- 边界 check：每个节点 dispatch 前
- within-node：worker 接到 Abort 后设置 CancelToken；sampler/streaming 内部循环 check
- LLM 额外：通过 WorkerClient 发 vLLM `AsyncEngine.abort(request_id)`
- 正在执行的节点不强杀 worker 子进程（强杀会泄漏 GPU 显存）

**故障路径**
- Worker crash → pipe EOF → mark in-flight task failed → 自动重启 worker
- Worker 重启会触发 resident model 重新 preload
- 同 group 排队中的 task 不受影响

**进度推送**
- 节点完成时 worker 通过 pipe 发 `progress(task_id, node, status)` 给主进程
- 主进程更新 ring buffer + 推 WebSocket；DB 只在 status 变化时 commit
- 复用现有 `/ws/workflow/{instance_id}`

### 2.3 跨 GPU group fallback

```
Worker-A (cuda:2) ──┐
                    ▼ (1) IPCRef 给主进程
              [Main Process]
                    │ (2) 检测 src 与 dst worker 不同 group
                    ▼
              host pinned memory 中转
                    │ (3) D→H → H→D
                    ▼
Worker-B (cuda:0)
```

96GB Pro 6000 → 24GB 3090 走 PCIe 4.0 ~25GB/s；典型 1024² latent 几 MB，开销 <1ms。可接受。

---

## 3. 数据结构 + 通信协议

### 3.1 数据库 schema 扩展

```python
class ExecutionTask(Base):
    # —— 已有（不动）——
    id: Mapped[int]
    workflow_id: Mapped[int | None]
    workflow_name: Mapped[str]
    status: Mapped[str]           # queued/running/completed/failed/cancelled
    nodes_total: Mapped[int]
    nodes_done: Mapped[int]
    current_node: Mapped[str | None]
    result: Mapped[dict | None]
    error: Mapped[str | None]
    duration_ms: Mapped[int | None]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    # —— V1.5 新增 ——
    priority: Mapped[int] = mapped_column(Integer, default=10)
    gpu_group: Mapped[str | None] = mapped_column(String(32), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    queued_at: Mapped[datetime | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    node_timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

Migration：所有新字段 nullable，一次 alembic add_column。

### 3.2 配置文件

#### `hardware.yaml`（新增）

```yaml
groups:
  - id: image
    gpus: [2]
    nvlink: false
    role: image
    vram_gb: 96

  - id: llm-tp
    gpus: [0, 1]
    nvlink: true
    role: llm
    vram_gb: 48

  - id: tts
    gpus: [3]
    nvlink: false
    role: tts
    vram_gb: 24

detection:
  mode: auto                 # auto / manual
```

`auto` 启动时跑 `nvidia-smi topo -m`，结果与本文件 diff 不一致就告警（不阻断）。`manual` 完全信任 yaml。

#### `models.yaml` 扩展

```yaml
flux2-dev:
  type: image
  local_path: "Flux2/Flux2-dev"
  vram_gb: 24
  resident: true
  preload_order: 10
  group_hint: image          # 推荐落 group，allocator 仍按 VRAM 选最优
  group_require: null

qwen3-35b:
  type: llm
  local_path: "Qwen/Qwen3-35B-A3B-GPTQ-Int4"
  vram_gb: 36
  resident: true
  preload_order: 20
  tensor_parallel_size: 2
  group_hint: llm-tp
  group_require: nvlink:true # 强约束：tp>1 必须 NVLink

cosyvoice2:
  type: tts
  local_path: "CosyVoice/CosyVoice2-0.5B"
  vram_gb: 4
  resident: false
  preload_order: null
  group_hint: tts
```

### 3.3 主进程 ↔ Worker 协议

走 `multiprocessing.Pipe` + msgpack 编码（dev 模式可 `NOUS_IPC_FORMAT=json` fallback 便于 journalctl 调试）。

```python
# 主进程 → worker
class LoadModel:
    kind: Literal["load_model"]
    model_key: str
    config: dict

class UnloadModel:
    kind: Literal["unload_model"]
    model_key: str

class RunNode:
    kind: Literal["run_node"]
    task_id: int
    node_id: str
    node_type: str
    model_key: str | None
    inputs: dict             # tensor 走 IPCRef
    is_deterministic: bool   # V1.5 仅落表，V1.6 缓存用

class Abort:
    kind: Literal["abort"]
    task_id: int
    node_id: str | None

class Ping:
    kind: Literal["ping"]

# worker → 主进程
class NodeResult:
    kind: Literal["node_result"]
    task_id: int
    node_id: str
    status: Literal["completed", "failed", "cancelled"]
    outputs: dict | None
    error: str | None
    duration_ms: int

class NodeProgress:
    kind: Literal["node_progress"]
    task_id: int
    node_id: str
    progress: float          # 0.0 ~ 1.0
    detail: str | None       # "step 12/30" / "247 tokens"

class ModelEvent:
    kind: Literal["model_event"]
    event: Literal["loaded", "unloaded", "load_failed"]
    model_key: str
    error: str | None

class Pong:
    kind: Literal["pong"]
    worker_id: str
    loaded_models: list[str]
```

### 3.4 CUDA IPC

```python
class IPCRef:
    storage_handle: bytes
    shape: tuple[int, ...]
    dtype: str
    device: str
```

接收方用 `torch.cuda.from_ipc_handle()` 还原。同 group 内 NVLink/PCIe P2P，跨 group fallback host pinned memory（见 §2.3）。

### 3.5 主进程内部数据结构

```python
@dataclass
class GroupScheduler:
    group_id: str
    queue: asyncio.PriorityQueue[QueuedTask]
    worker_client: WorkerClient
    cancel_events: dict[int, asyncio.Event]
    inflight_tasks: dict[int, asyncio.Task]

@dataclass(order=True)
class QueuedTask:
    sort_key: tuple[int, datetime]     # (priority, queued_at)
    task_id: int = field(compare=False)
    workflow_spec: dict = field(compare=False)

class TaskRingBuffer:
    """最近 200 条 task 快照，O(1) 读，给 TaskPanel / Dashboard 用。"""
    _items: collections.deque[TaskSnapshot]   # maxlen=200
    _by_id: dict[int, TaskSnapshot]
```

DB 是真相源（survives restart），ring buffer 是热缓存（survives request burst）。

### 3.6 NodeSpec

```typescript
interface NodeSpec {
  type: string
  inputs: NodeInput[]
  outputs: NodeOutput[]
  device?: string                       // ComfyUI MultiGPU 模式

  // V1.5 新增
  is_deterministic?: boolean            // V1.5 仅落表，V1.6 缓存用
  cancellable?: "boundary" | "within"   // within → 长循环内 check cancel_token
  estimated_duration_ms?: number        // 前端进度条平滑
}
```

### 3.7 WebSocket 事件词表（对齐 ComfyUI）

```typescript
type WSEvent =
  | { type: "execution_start"; task_id: number; workflow_name: string }
  | { type: "execution_cached"; task_id: number; node_id: string }    // V1.6
  | { type: "executing"; task_id: number; node_id: string }
  | { type: "progress"; task_id: number; node_id: string; value: number; max: number }
  | { type: "executed"; task_id: number; node_id: string; outputs?: object }
  | { type: "execution_error"; task_id: number; node_id: string; error: string }
  | { type: "execution_interrupted"; task_id: number; reason: string }
  | { type: "queue_changed"; group: string; pending: number; running: number }
```

---

## 4. 错误处理与隔离

设计原则：**故障域局限到 worker 子进程**。主进程永远不应因为某 worker crash 或某模型 OOM 而挂掉。

### 4.1 故障矩阵

| 故障类型 | 影响域 | 检测手段 | 恢复策略 |
|---|---|---|---|
| Worker 子进程 crash | 该 GPU group 全部任务 | pipe EOF / ping 超时 | 主进程重启 worker + resident 重新 preload |
| 模型 load OOM | 该模型 | adapter.load() 抛 CUDA OOM | LRU evict 同 group + 重试一次；二次失败标 load_failed |
| 模型 load 文件丢失/损坏 | 该模型 | adapter.load() 抛 FileNotFound | 标 `_load_failures[model_key]`，写 health，不重试 |
| vLLM 子进程启动失败 | 该 LLM | scanner health check 失败 | kill orphan，标 load_failed |
| 节点执行 OOM | 该节点 | adapter.run() 抛 CUDA OOM | 节点 fail；不重试；workflow 整体 fail |
| 节点超时 | 该节点 | asyncio.wait_for 超时 | 节点 fail + abort vLLM request；workflow fail |
| 节点 cancel | 该节点 + 后续 | cancel_event 触发 | 节点 cancelled；workflow status=cancelled |
| IPC pipe 阻塞 | 该 worker 当前 task | 写 pipe 5s 超时 | 视为 worker 假死 → SIGKILL + 重启 |
| DB 不可达 | 所有 task 持久化 | sqlalchemy 异常 | 主进程降级：继续执行 + ring buffer + WS；DB 恢复后下一次状态变化 commit |
| WebSocket 推送失败 | 仅前端实时性 | send 异常 | 静默吞掉，下次心跳重连 |
| Worker pipe 拥塞 | 该 worker 响应延迟 | queue size > 阈值 | 反压：scheduler 暂停向该 worker 派新 task |

### 4.2 Worker 生命周期

```
[main process startup]
  ├─ 读 hardware.yaml + 探测 → 决定要起几个 worker
  └─ for each group:
       fork worker subprocess
       wait worker 发 "ready" 消息（含 worker_id + GPU list）
       ▼
       [resident preload]
         按 preload_order 升序遍历 group_hint 匹配的 resident:true 模型
         send LoadModel → 等 ModelEvent(loaded/load_failed)
         load_failed 写 _load_failures[model_key] = error
         不阻断后续 preload，不阻断 API server start
```

#### Crash 检测 + 重启

```python
class WorkerSupervisor:
    PING_INTERVAL = 30
    PING_TIMEOUT = 10
    RESTART_BACKOFF = [5, 15, 60, 300]   # 封顶 5 min

    async def _watchdog(self):
        while True:
            await asyncio.sleep(self.PING_INTERVAL)
            try:
                await asyncio.wait_for(self.ping(), timeout=self.PING_TIMEOUT)
            except (asyncio.TimeoutError, BrokenPipeError):
                await self._restart()

    async def _restart(self):
        # 1. 终结 worker（SIGTERM 5s → SIGKILL）
        # 2. 当前所有 inflight task 标 failed (worker_crashed)
        # 3. backoff (防 crash 风暴)
        # 4. 重新 fork + preload resident
        # 5. 成功跑 30 min 后 reset crash_count
```

重启期间：该 group 的 PriorityQueue 不动，新 task 继续入队但 dispatcher 暂停。worker 起来后继续派发。

**Inflight task 处理策略**：全部标 failed (worker_crashed)，不重试。理由：不知道副作用是否已写（文件、token 计费），重试可能造成重复副作用。

### 4.3 OOM 处理

```python
async def get_or_load(self, model_key: str) -> LoadedModel:
    async with self._lock_for(model_key):
        if model_key in self._models:
            return self._models[model_key]
        for attempt in range(2):
            try:
                model = await self._load(model_key)
                self._models[model_key] = model
                return model
            except torch.cuda.OutOfMemoryError as e:
                if attempt == 0:
                    await self._evict_lru_in_group(needed_mb=self._estimate_mb(model_key))
                    continue
                self._load_failures[model_key] = f"OOM after evict: {e}"
                raise
            except Exception as e:
                self._load_failures[model_key] = str(e)
                raise
```

二次 OOM 后**用户手动 retry**：Dashboard 显示红色 banner，点 Retry Load 按钮触发重新加载。不做自动指数退避（避免用户调试时被反复 OOM 干扰）。

### 4.4 Cancel 双层

```python
# 节点边界（主进程 WorkflowExecutor）
async def execute_node(self, node, cancel_event):
    if cancel_event.is_set():
        raise NodeCancelled()
    return await self._dispatch_to_worker(node, cancel_event)

# within-node（worker 内 adapter）
class CancelToken:
    def __init__(self, task_id: int, worker_pipe):
        self._task_id = task_id
        self._cancelled = False
    def is_cancelled(self) -> bool:
        return self._cancelled
    def raise_if_cancelled(self):
        if self._cancelled:
            raise NodeCancelled()

# image sampler 用法
for step in range(num_inference_steps):
    cancel_token.raise_if_cancelled()
    latents = self._denoise_step(latents, step)

# LLM streaming 用法
async for token in vllm_stream:
    if cancel_token.is_cancelled():
        await vllm_engine.abort(request_id)
        raise NodeCancelled()
    yield token
```

TTS / VAE / 短节点只在边界 check（计算太短，within-node check overhead 反而大）。

### 4.5 隔离边界

```
┌────────────────────────────────────────────────────────┐
│ Main Process                                           │
│  - Event Loop A                                        │
│  - DB session pool                                     │
│  - WorkerSupervisor × N                                │
│  - GroupScheduler × N (per-group PriorityQueue)        │
│  - cancel_events: dict[task_id, asyncio.Event]         │
└────────────────────────────────────────────────────────┘
       │ multiprocessing.Pipe (msgpack)
       │ pipe EOF/timeout 主进程能感知，worker 死不连累主进程
       ▼
┌────────────────────────────────────────────────────────┐
│ Worker Subprocess (3 instances)                        │
│  - Event Loop B (独立)                                  │
│  - ModelManager (worker 内单例)                         │
│    - per-model asyncio.Lock                            │
│    - LRU evict 只在本 worker 管的 GPU group 内做         │
│  - vLLM AsyncEngine (LLM worker 内)                     │
│    - 自身又是 subprocess，crash 由 worker 重启它         │
└────────────────────────────────────────────────────────┘
```

关键性质：
- 主进程 DB session / HTTP handler / WebSocket 跑 Event Loop A，worker crash 不影响
- Worker 内某 model OOM 不影响同 worker 其他 model（per-model lock + GPU 内存隔离）
- Worker 内某 model native segfault 拖死整个 worker → 走 §4.2 restart

### 4.6 DB 不可达降级

DB 写入失败仅记 log，业务继续执行，靠 ring buffer + WebSocket 推送给前端。DB 恢复后下一次状态变化时 commit。代价：DB 故障期间 ring buffer 容量（200）外的历史会丢。

理由：用户场景是单 admin 推理 infra，DB 短暂不可达不应中断 LLM 推理流。一致性可以靠 ring buffer + DB 双写在恢复期 reconcile。

### 4.7 关键边界 case

1. **Cancel 已 dispatch 但 worker 还没收到**：主进程标 task=cancelled，等 NodeResult 回来直接丢弃
2. **多个 task 同时 cancel**：cancel_events 是 dict[task_id]，互不影响
3. **API server 重启**：启动时扫 DB：
   - status=running → 标 failed (server_restarted)
   - status=queued → 重新入对应 group queue
4. **某 group 队列堆积 >1000**：拒绝新入队，返回 503 + Retry-After

---

## 5. 测试策略

### 5.1 测试分层

```
┌─────────────────────┐
│ E2E (真 GPU)         │  ~10 个，dev box 跑，CI skip
│ pytest -m e2e        │
├─────────────────────┤
│ Integration (mock    │  ~30 个，CI 默认跑
│   worker subprocess) │
│ pytest -m integration│
├─────────────────────┤
│ Unit (纯主进程逻辑)   │  ~80 个，CI 默认跑
│ pytest               │
└─────────────────────┘
```

### 5.2 Unit 测试覆盖（无 GPU、无 subprocess）

| 模块 | 测试要点 |
|---|---|
| `GroupScheduler` | 优先级排序、同优先级 FIFO、队列堆积 503、cancel_event 在 dispatch 前后 |
| `WorkerSupervisor.RESTART_BACKOFF` | crash_count 累加 + reset；exponential backoff 序列 |
| `TaskRingBuffer` | maxlen=200 evict 最旧、by_id lookup、list_recent limit |
| `hardware.yaml` 探测 | mock `nvidia-smi topo -m`；yaml override 覆盖探测 |
| `preload_order` 排序 | resident:true + preload_order 升序；null 排最后 |
| NodeSpec 校验 | tp>1 强制 nvlink:true；device 字段格式 |
| IPCRef 序列化 | msgpack 编码/解码往返；dev 模式 JSON fallback |
| Cancel 双层 check | 边界路径 + within-node CancelToken |

### 5.3 Integration 测试（mock worker subprocess）

主进程 + fake worker（纯 Python，不加载真模型），跑通完整 IPC 协议。

| 场景 | 验证 |
|---|---|
| Workflow 完整生命周期 | enqueue → dispatch → run_node → completed；DB + ring buffer + WS |
| 优先级抢占 | batch task A 先入，interactive B 后入，B 先 dispatch |
| Cancel inflight | dispatch 后 cancel → fake worker 模拟 Abort → status=cancelled |
| Worker crash 检测 | fake worker kill self → PING_TIMEOUT 内检测 → inflight 标 failed |
| Worker 重启 resident preload | 重启后收 LoadModel 序列；preload_order 升序 |
| 模型 load_failed 不阻断 | 某 model_key 回 load_failed → 后续继续；/health 暴露 failure |
| DB 不可达降级 | mock SQLAlchemy OperationalError → ring buffer + WS 正常 |
| 队列堆积 503 | 灌 1001 个 task → 第 1001 个 503 + Retry-After |
| 跨 group 节点 | image + tts workflow → 两 group 并发；跨 worker tensor host pinned fallback |
| API server 重启恢复 | DB 中 status=running → failed (server_restarted)；queued → 重新入队 |

CI 跑 `pytest -m integration`，5 分钟内完成。

### 5.4 E2E 测试（真 GPU，dev box）

每次 PR 合并前手动跑。

| 场景 | 验证 |
|---|---|
| Flux2 + CosyVoice2 并发 | 真双卡 utilization 同时拉满 |
| vLLM tp=2 | Qwen3-35B 在 GPU 0+1 NVLink pair 上 load；nvidia-smi 看 P2P 流量 |
| Image sampler within-node cancel | 30 step sampler，第 5 步 cancel → <500ms 停 |
| OOM evict 路径 | 故意 load 超 VRAM → LRU evict + 重试；二次 OOM → load_failed |
| Worker 真 crash 恢复 | `kill -9` worker → 30s 内重启 + resident 重 load |
| 主进程重启幂等 | 已有 vLLM orphan → reconnect healthy / kill unhealthy（回归） |

### 5.5 故障注入（`tests/chaos/`，每周手动跑）

```python
# test_worker_crash_storm.py
async def test_worker_repeated_crashes():
    """连续 5 次 worker crash → 验证 backoff + 主进程不挂"""

# test_pipe_slow_consumer.py
async def test_worker_blocks_pipe():
    """fake worker 不读 pipe → 主进程 5s 写超时 + 反压"""

# test_db_flaky.py
async def test_db_intermittent_failures():
    """50% 概率 DB OperationalError，soak 1000 task → ring buffer 一致"""
```

### 5.6 测试基础设施

| 已有 | 复用 |
|---|---|
| `tests/conftest.py` 强制 `ADMIN_PASSWORD=""` | 复用 |
| `NOUS_DISABLE_FRONTEND_MOUNT=1` | 复用 |
| `pytest-asyncio` | 复用 |

| 新增 | 用途 |
|---|---|
| `tests/fixtures/fake_worker.py` | mock worker subprocess，可配置 crash/slow/fail-load |
| `tests/fixtures/hardware_topo.py` | mock `nvidia-smi topo -m` 输出 |
| `pytest.ini` markers | `e2e`, `integration`, `chaos` |

---

## V1.5 vs V1.6 节点输出缓存取舍

ComfyUI 的最大优势之一是**节点输出缓存**：按节点输入 hash 缓存输出，重跑同 workflow 只改 seed 时前面的 CLIP encode、VAE decode 等节点全 hit cache，几秒内完成。但 V1.5 暂不实现，理由如下。

### 为什么 V1.5 不做

1. **架构基础不稳就上层优化**会埋雷：当前 inference race / 调度 / OOM 三件事还没解决，缓存只是性能加速。
2. **正确性门槛高**：缓存命中 = 必须保证"相同输入 + 相同 model 版本 + 相同节点代码"产出相同输出。涉及输入 hash 标准化、`is_deterministic` 强制声明、model 版本指纹。做不对会出"明明改了参数还看到旧结果"的脏 bug。
3. **存储语义未定**：缓存放 worker 进程内存？跨 worker 共享？容量怎么淘汰？V1.5 的 worker 拓扑刚刚确定，先观察使用模式再决定。

### V1.5 已经做的前置准备

- NodeSpec 加 `is_deterministic: bool` 字段（仅落 ExecutionTask.node_timings，不影响行为）
- WSEvent 词表预留 `execution_cached` 类型
- `node_timings` JSON 字段为每节点保留 `cached: bool`（V1.5 永远 false）

### V1.6 实现条件

实现节点输出缓存前必须先解决：

1. **输入 hash 标准化**：每种 NodeInput 类型必须有 deterministic hash 方法（包括 tensor、dict、file path）
2. **`is_deterministic` 强制声明**：所有 NodeSpec 必须显式声明；缺失视为 false（不缓存）
3. **Model 版本指纹**：models.yaml 加 `version_hash`（基于 weight file mtime + size），缓存 key 包含此字段
4. **存储拓扑决策**：单 worker 内内存 LRU（最简单）vs 跨 worker 共享磁盘缓存

V1.6 设计将单独写 spec doc。

---

## 实施分 Lane

按依赖顺序拆 PR：

| Lane | 内容 | 依赖 |
|---|---|---|
| A | hardware.yaml + topo 探测 + GPUAllocator 重构 | 无 |
| B | execution_tasks schema migration + TaskRingBuffer | 无 |
| C | WorkerSupervisor + 子进程框架（fake adapter，跑通 IPC） | A |
| D | ModelManager 迁入 worker（image worker 先迁，验证 inference lock） | B, C |
| E | LLM worker 迁入（vLLM tp + group_require:nvlink） | D |
| F | TTS worker 迁入 | D |
| G | GroupScheduler + priority + cancel 双层 | C |
| H | resident preload_order + load_failures + /health 扩展 | D |
| I | TaskPanel 前端扩展 + Dashboard worker 标识 | B, G |
| J | Integration + chaos 测试套 | C 到 H 全部 |

每 Lane 一个 PR，本地 ruff + tsc + vite build 过 → push → CI 绿 → merge。

---

## Open Questions

无。本 spec 决策点全部锁定。

---

## REVIEW IN PROGRESS — plan-eng-review 2026-05-14（待重写并入正文）

`/plan-eng-review` 审查中。下列决策已获用户确认，**尚未并入上面的正文**，下次会话需据此重写 spec。

- **D1 复杂度门禁** — 按 10-Lane 全量推进，设计不拆、实施不拆。
- **D2 (A1) CUDA IPC** — 删除 CUDA IPC / IPCRef / from_ipc_handle。每 group 恰好 1 worker → 跨 worker 永远跨 group → P2P 路径是死代码。跨 worker tensor 一律 host pinned memory 中转。§3.3/§3.4 重写。
- **D3 (A2) 调度器整合** — 新增前置 **Lane 0**：删除 `model_scheduler.py`，`monitor.py` + `gpu_monitor.py` 改用 `model_manager`；盘点 `src/gpu/model_manager.py` 职责后合并或删除。零行为变化的独立 refactor PR，先于所有 V1.5 Lane。
- **D4 (A3) 硬件错配** — spec 保持 N-group 通用（不写死 3 worker），启动按探测/yaml 动态决定 worker 数。新增「当前 2 卡部署」一节，明确画出 2×3090 的 group 布局并选定取舍（NVLink 对优先 tp-LLM vs 拆开优先并发）。`hardware.yaml` 发两份示例（2 卡当前 / 3 卡未来）。
- **D5 (A4) 命名冲突** — spec 的「Worker 子进程」概念全部改名 **GPU Runner**：WorkerSupervisor→RunnerSupervisor，WorkerClient→RunnerClient，worker_id→runner_id。`src/workers/` 不动。
- **D6 (A5) 两类 runner** — 区分：image/TTS runner 走 per-group 串行队列（一次一个 GPU job）；LLM runner 是 vLLM 的并发代理（不串行化）。新增一节枚举所有 inline 执行点（openai/anthropic/ollama compat + responses.py + workflow_executor）的改道方式。
- **D7 (A6) DB reconcile** — 设计明确机制：ring buffer 每条 TaskSnapshot 加 `db_synced: bool`，DB 写失败置 false，后台任务检测 DB 恢复后遍历 ring buffer 批量补写。
- **D8 (A7) LLM Runner 边界** — LLM Runner 只管 vLLM 生命周期（spawn/health/preload/abort/OOM 重启）；主进程 compat 路由直连 vLLM HTTP 端口，零 per-token pipe 开销。
- **D9 (C3) Runner 内部并发** — spec §4.4 补 runner 内部并发设计：runner 内跑两个 asyncio task——pipe-reader（持续读消息，收 Abort 置位 CancelToken）+ node-executor（跑 adapter.run，长循环内轮询 CancelToken）。adapter.run 须为 async 可让出。
- **D10 (C4) 混合节点 workflow** — workflow executor 留主进程：llm 节点 → executor 直接 HTTP 调 vLLM（与 compat 路由同路）；image/TTS 节点 → dispatch 到对应 runner 串行队列。workflow 分组逻辑区分「dispatch 节点」与「inline HTTP 节点」。
- **D11 (P1) 大载荷传输** — image 节点结果 runner 直接写 `outputs/{task_id}/`（复用旧 spec 的 output_dir），pipe 只传路径 + 元数据；跨节点中间 tensor 走 host pinned；小标量/dict 走 msgpack pipe。

### Outside voice（codex 配置坏，回退 Claude 子代理）翻出的 cross-model tension，已用户确认

- **D13 (战略) 维持 D1** — outside voice 主张「证明设计的硬件没到，子进程架构今天无法验证」，建议重开 D1。用户重新确认：维持 D1，10-Lane 全量现在全建，Pro 6000 到了即用。
- **D14 (G1/G2) within-node cancel 含 adapter 重写** — `adapter.run` 现在不是 async-cancellable（`image_diffusers.py` 把扩散 pipe 当 `to_thread` 里一个不透明阻塞调用）。V1.5 包含 adapter 重写：image adapter 接 diffusers `callback_on_step_end`，callback 在 worker 线程里检查 `threading.Event`，pipe-reader 收到 Abort 就 set。`asyncio.wait_for` 超时路径也走同一个 callback flag（`wait_for` 取消 `to_thread` 不会停 CUDA kernel）。
- **D15 (F3) topo 探测 manual-only** — V1.5 不解析 `nvidia-smi topo -m`。`hardware.yaml` 是唯一真相源，手写。auto 探测推迟到 Pro 6000 到货（届时有 2 套硬件可验证解析器）。`detection.mode` 字段从 spec §3.2 删除。
- **D16 (G3) 接受罕见双重故障窗口** — §2.2「DB enqueue 前 commit」与 D7「DB 不可达 fail-soft」的恢复路径矛盾：spec §4.7 明写「DB 在 enqueue 时就不可达 + 主进程在 DB 恢复前重启」这个双重故障下该 task 会丢失。不建 durable spool（本机 Postgres + systemd 同机，DB 独立不可达极罕见，为它建持久化层是 over-engineering）。
- **D17 (S3) workflow `/run` 端点改纯异步** — `/v1/workflows/{id}/run` 立即返回 202 + task_id，客户端轮询 `/v1/tasks/{id}` 或订阅 WS 看进度/结果（SeedDance 模式）。mediahub 等上游调用方迁移到新契约。注意：单次 LLM 调用的同步体验不受影响——compat 路由按 D6/D8 直连 vLLM HTTP，本来就同步、不进队列。D17 只改多节点 workflow 端点。新增一节说明 `/run` 契约变更 + 上游迁移指引。

### 重写时还需并入（review 翻出、无 A/B/C 备选的强制项）

- **G4** — Lane 0 删 `model_scheduler.py`，但它持有 `get_llm_base_url()`（`model_scheduler.py:233`），正是 D8/D10 依赖的「主进程直连 vLLM HTTP」接缝。Lane 0 必须在删除前把这个 URL 查找重新安置（迁入合并后的 model_manager）。
- **G5** — `services/model_manager.py`（`asyncio.Lock`）与 `src/gpu/model_manager.py`（`threading.Lock` + `VRAMTracker`）不是重叠是互补：后者有 NVLink allocator 需要的真实 VRAM 核算（`can_load` / `VRAMTracker`）。Lane 0「合并或删除」必须真正设计这个合并（asyncio-locked 类 + thread-locked 类 → 跑自己 event loop 的 runner 子进程内），不是 hand-wave 的 refactor。
- **F1** — `multiprocessing.Pipe` 对象不可 await：D9 的 pipe-reader asyncio task 需要 `loop.connect_read_pipe` / `add_reader` / 线程桥。`Pipe.send` 无 timeout 参数——§4.1「IPC pipe 阻塞 5s 超时」需写线程或非阻塞 fd。spec §3.3 须明确这个实现约束。
- **F2** — §4.2 worker 重启 re-preload 前需加「等 nvidia-smi 显示 GPU free」的 gate：OOM 相关 native fault 后死进程的 CUDA context 回收是异步的，backoff `[5,15,60,300]` 可能太短。
- **S3** — 新增一个 Lane 显式拥有 `workflow_executor.py` 重写（`execute()` 现在是 `workflow_executor.py:102` 一个扁平顺序循环，D10 要拆成 dispatch 节点 vs inline-HTTP 节点）。`execute_workflow_direct`（`workflows.py:142`）的 inline 执行 + D17 的契约变更都在这个 Lane。

- **C1** — §3.5/§4.3 改为「复用 model_manager 已有的 `_load_failures` dict + `evict_lru(gpu_index)`」，不要重新发明。
- **C2** — §3.3 IPC 协议整节重写：删 IPCRef；RunNode 只剩 image/TTS；保留 LoadModel/UnloadModel/Abort/Ping。
- **C6** — §3.2 提一句现有 `_detect_vllm_gpus*` 与 hardware.yaml topo 探测的关系（整合或并存）。
- **§5 测试** 需补 10 项，其中 4 项 CRITICAL 回归：
  1. [回归] Lane 0 后 monitor.py / gpu_monitor.py 仍正确报告加载状态 + idle-TTL 卸载仍生效
  2. [回归] A5 后 4 个 compat 路由仍产出正确输出
  3. [回归] workflow inline → queued（D17 改纯异步 202）后执行结果不变
  4. [回归] src/gpu/model_manager.py 合并/删除后所有调用点仍工作
  5. LLM lifecycle runner 不串行化 vLLM 请求
  6. runner 内 pipe-reader + executor task 分离；Abort-during-node-execution within-node 生效
  7. host-pinned 跨 worker tensor 传递
  8. DB 恢复后 reconcile 路径：db_synced=false 批量补写
  9. 混合节点 workflow：image dispatch + llm inline HTTP
  10. [D14] image adapter `callback_on_step_end` 接入 + cancel/timeout 信号穿过 to_thread 边界（threading.Event）

### review 剩余步骤（下次会话继续）

- [x] review 四个 section + outside voice 全部完成，D1–D17 全部确认
- [ ] 必需输出：NOT in scope / What already exists / TODOS.md updates / Failure modes / Worktree 并行化策略 / Completion summary（本次会话已在对话中给出，重写时并入 spec）
- [ ] 据 D1–D17 + G4/G5/F1/F2/S3 + §5 重写 spec 正文，删除本「REVIEW IN PROGRESS」节
- [ ] Review log + Readiness Dashboard + Plan file report
- [ ] 用户审核重写后的 spec → 转 `superpowers:writing-plans`

## References

- `docs/superpowers/specs/2026-03-25-global-task-management-design.md` — V1 ExecutionTask 表（本 spec 扩 schema）
- `docs/superpowers/specs/2026-04-10-gpu-process-management-design.md` — V1 orphan handling + kill endpoint
- ComfyUI `execution.py` + `model_management.py` — 节点缓存、smart memory free、WS 事件词表的灵感来源
- ComfyUI-MultiGPU — Loader 节点 device 字段的设计参考

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | ISSUES_FOUND (claude subagent) | 11 findings, 4 → cross-model tension, all resolved D13–D17 |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN (PLAN) | 13 issues (7 arch + 1 quality-decisions + 1 perf + 4 critical regressions), 2 critical gaps; 18 decisions confirmed, rewrite pending |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE VOICE:** codex 配置损坏（`tui.alternate_screen`），回退 Claude 独立子代理。翻出 G1/G2（adapter 非 async-cancellable）、G3（DB 恢复矛盾）、G4（`get_llm_base_url` 接缝）、G5（两个 ModelManager 互补非重叠）、F1/F2/S3，战略挑战 D1。
- **CROSS-MODEL:** 4 处 tension（D1 scope / within-node cancel / topo 探测 / DB reconcile）全部经 AskUserQuestion 由用户裁决（D13–D17）。
- **UNRESOLVED:** 0（18 项决策全部确认）。
- **VERDICT:** NOT CLEARED — 18 项决策已确认但尚未并入 spec 正文；2 个 CRITICAL GAP（G2 to_thread cancel 泄漏、F1 Pipe 无 timeout）须在重写时设计落地。重写完成 + 并入正文后方可 CLEARED。
