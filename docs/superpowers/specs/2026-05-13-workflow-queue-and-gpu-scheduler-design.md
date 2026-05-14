# Workflow Queue + GPU Scheduler (V1.5) Design

**Author:** heygo
**Date:** 2026-05-13
**Status:** Reviewed (plan-eng-review + plan-design-review 决策已并入，pending implementation plan)

## Problem

V1 起 image / TTS / LLM 全在 API server 主进程内串行执行，存在三类问题需要**同时**解决：

1. **并发竞态** — `image_diffusers.generate` 等节点没有 inference-level lock，并发请求会撞同一 `adapter.pipe`，输出错乱甚至 CUDA OOM 拖死整个进程。
2. **GPU 分配粗放** — 当前 model_manager 只看 `gpu_index`，不感知 NVLink 拓扑；vLLM tensor-parallel 模型有可能落到非 NVLink 的卡对上，PCIe 通信成瓶颈。
3. **多模型 resident 缺策略** — 启动时按 yaml 顺序一个一个 load，某个 OOM 阻断后面的，API server 起不来。

V1.5 把这三件事放在**一个架构**下解决：进程隔离（GPU Runner 子进程）+ 拓扑感知调度 + 鲁棒 preload。

### 硬件背景

| 卡 | VRAM | NVLink | 角色 |
|---|---|---|---|
| GPU 0 (3090) | 24 GB | 与 GPU 1 配对 | LLM tensor-parallel |
| GPU 1 (3090) | 24 GB | 与 GPU 0 配对 | LLM tensor-parallel |
| GPU 2 (Pro 6000 Blackwell) | 96 GB | 单卡 | image |
| (可选 GPU 3) | 24 GB | 单卡 | TTS |

CPU AMD Threadripper 7965WX 24C/48T，93 GB RAM。

设计**对 group 数通用**（N-group），不写死 worker/runner 数量。Pro 6000 尚未到货，当前实际是 2×3090 部署，见 [§1.4 当前 2 卡部署](#14-当前-2-卡部署)。

### 不在 V1.5 范围

- 节点输出缓存（按输入 hash skip 已算过的节点）—— 见 [V1.6 取舍](#v15-vs-v16-节点输出缓存取舍)
- 跨机分布式（永远不做，本机推理 infra）
- 用户 quota / rate limit（单 admin，不需要）
- 模型自动下载/管理（已有独立路径，不重做）
- `nvidia-smi topo -m` 自动拓扑探测（V1.5 manual-only，见 D15 / §3.2）
- CUDA IPC / P2P 跨卡张量直传（每 group 恰好 1 runner，跨 runner 永远跨 group，P2P 是死代码，见 D2 / §3.4）

## Goals

1. 同模型/同 GPU 任务**串行**执行，不同 GPU group 任务**并发**执行
2. NVLink-aware 分组（manual `hardware.yaml`）；vLLM tp 模型只能落 NVLink:true group
3. GPU Runner crash 不拖死主进程；resident preload OOM 不阻断 API server 启动
4. ExecutionTask 历史可观测：queued/started/finished 时间、runner_id、gpu_group、node_timings
5. 前端 TaskPanel 重构为 Buildkite 风 per-runner 泳道 + Dashboard 标识
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
| 进程模型 | 主进程 asyncio 调度 + 每 GPU group 一个 **GPU Runner** 子进程 |
| Runner 拓扑 | N-group 通用，启动按 `hardware.yaml` 动态决定 runner 数 |
| 两类 Runner | image/TTS runner = per-group 串行 PriorityQueue；LLM runner = 只管 vLLM 生命周期，不串行化请求 |
| GPU 分组 | `hardware.yaml` 手写，唯一真相源（V1.5 不解析 `nvidia-smi topo -m`） |
| 任务粒度 | Workflow 入队为单 task；executor 在主进程内按节点 GPU 需求分组 |
| 优先级 | 2 级（interactive=0 > batch=10），同级 FIFO |
| Cancel | 节点边界 + within-node（sampler step / LLM token），含 adapter 重写 |
| 跨 GPU 数据传输 | 一律 host-pinned memory 中转（无 CUDA IPC） |
| 大载荷 | image 结果 runner 写 `outputs/{task_id}/`，pipe 只传路径 + 元数据 |
| Inference lock | image/TTS runner 内 per-model `asyncio.Lock`，wrap 整个 `adapter.run()` |
| History | `execution_tasks` 扩 schema + 主进程 ring buffer（200 entries，`db_synced` 标志） |
| Resident 启动 | `preload_order` 数字升序，失败 fail-soft，暴露到 `/health` |
| 协议层 | `multiprocessing.Pipe` + msgpack（dev 模式 JSON fallback） |
| Workflow `/run` | 纯异步：返回 202 + task_id，客户端轮询 / 订阅 WS |
| Frontend UX | TaskPanel 重构为 Buildkite 风 runner 泳道 + 完成通知 + 响应式 |

### 1.2 模块拓扑

```
┌────────────────────────────────────────────────────────────────────┐
│  Main Process (API server, asyncio loop)                           │
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ FastAPI     │  │ Scheduler    │  │ WorkflowExecutor         │  │
│  │ /v1/...     │─▶│ - per-group  │─▶│ - 解析节点 DAG           │  │
│  │ compat 路由 │  │   PriorityQ  │  │ - dispatch 节点 vs       │  │
│  │   ↓直连vLLM │  │ - ring buf   │  │   inline-HTTP 节点       │  │
│  └─────────────┘  └──────────────┘  └──────────────────────────┘  │
│         │                │                     │                   │
│         │                ▼                     ▼                   │
│         │       ┌─────────────────────────────────────────────┐   │
│         │       │ RunnerClient (主进程侧，仅 image/TTS)        │   │
│         │       │  - asyncio bridge ↔ multiprocessing pipe    │   │
│         │       │  - 节点级 RPC: run_node(spec, inputs)       │   │
│         │       └─────────────────────────────────────────────┘   │
│         │              │              │                            │
│         ▼              ▼              ▼              │ HTTP        │
│  ┌──────────┐   ┌──────────┐  ┌──────────┐           ▼直连        │
│  │ Postgres │   │ Runner-I │  │ Runner-T │   ┌──────────────┐     │
│  │ tasks +  │   │ (image)  │  │ (tts)    │   │ Runner-L     │     │
│  │ schema   │   │ Pro 6000 │  │ GPU 3    │   │ (llm)        │     │
│  └──────────┘   ├──────────┤  ├──────────┤   │ GPU 0+1 NVL  │     │
│                 │ 串行 PQ  │  │ 串行 PQ  │   ├──────────────┤     │
│                 │ per-model│  │ per-model│   │ 只管 vLLM    │     │
│                 │ lock     │  │ lock     │   │ subprocess   │     │
│                 │ Flux2/SD │  │ Cosy2/.. │   │ 生命周期     │     │
│                 └──────────┘  └──────────┘   └──────────────┘     │
└────────────────────────────────────────────────────────────────────┘
```

两类 Runner 本质不同：
- **image/TTS Runner** — 一次跑一个 GPU job，主进程 RunnerClient 通过 pipe dispatch 节点，runner 内 per-group 串行 PriorityQueue。
- **LLM Runner** — 只负责 vLLM 子进程的 spawn / health / preload / abort / OOM 重启，**不串行化推理请求**。主进程 compat 路由和 workflow executor 的 llm 节点都直连 vLLM HTTP 端口（见 D8 / §1.5）。

### 1.3 三件事如何统一解决

- **并发竞态** → image/TTS 的 per-group PriorityQueue 解决"同 GPU 任务串行"；runner 内 per-model lock 解决"同模型 adapter race"。LLM 的并发由 vLLM 自身的 continuous batching 处理，不需要外层串行化。
- **NVLink-aware 调度** → `hardware.yaml` 手写产出 `groups[]`；allocator 按 group 分配；tp 模型在 spec 校验层强制 `group_require: nvlink:true`。
- **Resident preload** → ModelManager 启动钩子按 `preload_order` 升序 dispatch；fail-soft 写 `_load_failures`，暴露到 `/health`。

### 1.4 当前 2 卡部署

Pro 6000 未到货，当前硬件 = 2×3090（NVLink 配对）。group 布局存在取舍：

| 方案 | group 布局 | 取舍 |
|---|---|---|
| A（选定）| 单 group `llm-tp` = GPU [0,1] NVLink；image/TTS 与 LLM 时分复用同卡 | tp-LLM 走 NVLink 高带宽；image/TTS 任务进 LLM group 队列，与 LLM 推理时分复用，串行但不撞车 |
| B（不选）| 拆成 image=GPU0 / llm=GPU1 单卡 | image/LLM 可并发，但 LLM 失去 tp=2 与 NVLink，35B 模型放不下单卡 |

**选 A**：当前主力负载是 LLM，35B GPTQ 模型必须 tp=2 + NVLink 才放得下；image/TTS 是次要负载，可接受与 LLM 时分复用。Pro 6000 到货后切回 3-group 独立布局，image 独占 96GB 卡。

`hardware.yaml` 提供两份示例（见 §3.2）：`hardware.2gpu.yaml`（当前）和 `hardware.3gpu.yaml`（未来）。

---

## 2. 数据流

### 2.1 完整路径（workflow 提交 → 完成）

```
[API/UI] ──POST /v1/workflows/{id}/run──▶ [Scheduler]
  1. 解析 workflow → 收集模型依赖
  2. 计算所需 GPU groups
  3. 创建 ExecutionTask (status=queued, queued_at=now)
  4. DB commit → 推入对应 group 的 PriorityQueue
  5. 立即返回 202 + task_id（D17 纯异步契约）
                                              ▼
                                         [Group Dispatcher]
  6. 弹出 (priority, queued_at, task_id) 最小者
  7. 标记 status=running, started_at=now
                                              ▼
                                         [WorkflowExecutor]（主进程内）
  8. DAG 拓扑排序
  9. 按节点类型分流：
     - llm 节点    → executor 直接 HTTP 调 vLLM（inline，不进 runner 队列）
     - image/TTS   → dispatch 到对应 runner 串行队列
     - 不同 group 节点 asyncio.gather() 并发
                                              ▼
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
              [RunnerClient.run_node(spec)]              [HTTP → vLLM :port]
  10a. 序列化输入（大 tensor host-pinned，         10b. OpenAI-compat 请求
       小标量/dict msgpack）→ pipe → runner              直连 vLLM HTTP
  11a. ModelManager.get_or_load（per-model lock）  11b. vLLM continuous batching
  12a. adapter.run(inputs, cancel_token)
       image 结果写 outputs/{task_id}/
  13a. 返回 NodeResult（路径 + 元数据）
                          └───────────────────┬───────────────────┘
                                              ▼
                                         [WorkflowExecutor]
  14. nodes_done++, current_node 更新
  15. WebSocket 推 progress 事件
                                              ▼
                                         (所有节点完成)
  16. status=completed, finished_at, duration_ms
  17. 写 result, node_timings JSON
  18. ring buffer push（db_synced 标志）
  19. invalidate cache: ["tasks"]
```

### 2.2 时序与一致性

**入队阶段（主进程）**
- ExecutionTask 在 DB commit 之后才入 PriorityQueue —— 保证宕机重启可恢复（启动时扫 status=queued 的 task 重新入队）
- 入队 sort key = `(priority, queued_at, task_id)` —— 同 priority 内 queued_at FIFO
- DB 在 enqueue 时不可达的故障窗口见 [§4.7](#47-关键边界-case)

**节点分流阶段（主进程 WorkflowExecutor）**
- 节点 spec `type` → 判定 dispatch 节点（image/TTS）还是 inline-HTTP 节点（llm）
- dispatch 节点：解析 `device` → GPU group id → 投到对应 runner 串行队列
- inline-HTTP 节点：executor 直接 HTTP 调 vLLM（与 compat 路由同路）
- 同 group dispatch 节点串行 await，不同 group `asyncio.gather()` 并发
- 跨节点 tensor 输出：同 runner → 直接 dict 传递；跨 runner → host-pinned memory 中转（无 IPC）

**image/TTS Runner 内部**
- ModelManager 每 runner 独立实例（不再是模块全局单例）
- per-model `asyncio.Lock` wrap 整个 `adapter.run()`
- LRU evict 只在本 runner 管的 GPU group 范围内做
- runner 内并发模型见 [§4.4](#44-runner-内部并发--cancel-双层)

**LLM Runner 内部**
- 只持有 vLLM 子进程句柄，监控 health、执行 preload / abort / OOM-restart
- 不接 RunNode 消息，不串行化推理；推理请求由主进程直连 vLLM HTTP

**Cancel 路径**
- `POST /v1/tasks/{id}/cancel` → 主进程 scheduler 设置 task.cancel_event
- 边界 check：每个节点 dispatch 前
- within-node（image/TTS）：runner 的 pipe-reader 收到 Abort → set `threading.Event` → adapter callback 检查
- within-node（LLM）：executor / compat 路由发 vLLM HTTP abort（`/v1/...` cancel 或 `AsyncEngine.abort`）
- 正在执行的节点不强杀 runner 子进程（强杀会泄漏 GPU 显存）

**故障路径**
- image/TTS Runner crash → pipe EOF → mark in-flight task failed → 自动重启 runner
- LLM Runner crash → vLLM 也随之失联 → 重启 runner → 重新 spawn vLLM + preload
- Runner 重启会触发 resident model 重新 preload（gate 见 §4.2）
- 同 group 排队中的 task 不受影响

**进度推送**
- image/TTS：节点完成时 runner 通过 pipe 发 `NodeProgress` / `NodeResult`
- LLM：executor 从 vLLM HTTP stream 直接读 token，自行折算 progress
- 主进程更新 ring buffer + 推 WebSocket；DB 只在 status 变化时 commit
- 复用现有 `/ws/workflow/{instance_id}`

### 2.3 跨 GPU group 数据传输

每 group 恰好 1 runner，跨 runner 永远跨 group。无 CUDA IPC / P2P 路径，一律 host-pinned 中转：

```
Runner-I (cuda:2) ──┐
                    ▼ (1) D→H：cudaMemcpy 到 host-pinned buffer
              [Main Process]
                    │ (2) 经 pipe 传 buffer 引用（或 shared memory）
                    ▼
              host-pinned memory 中转
                    │ (3) H→D：拷到目标 runner GPU
                    ▼
Runner-L / Runner-T
```

96GB Pro 6000 → 24GB 3090 走 PCIe 4.0 ~25GB/s；典型 1024² latent 几 MB，D→H→D 开销 <1ms。可接受。小标量/dict 直接走 msgpack pipe，不过 host-pinned。

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
    runner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    queued_at: Mapped[datetime | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    node_timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

Migration：所有新字段 nullable，一次 alembic add_column。

### 3.2 配置文件

#### `hardware.yaml`（新增，V1.5 manual-only，唯一真相源）

V1.5 **不解析** `nvidia-smi topo -m`，拓扑全靠手写。auto 探测推迟到 Pro 6000 到货（届时有 2 套硬件可验证解析器）。无 `detection.mode` 字段。

`hardware.3gpu.yaml`（未来，Pro 6000 到货后）：

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
```

`hardware.2gpu.yaml`（当前，见 §1.4 方案 A）：

```yaml
groups:
  - id: llm-tp
    gpus: [0, 1]
    nvlink: true
    role: llm            # image/TTS 节点也落此 group，与 LLM 时分复用
    vram_gb: 48
```

启动按 yaml 的 `groups[]` 数量动态决定起几个 runner，不写死。

现有 `_detect_vllm_gpus*`（vLLM 启动时用的 GPU 选择）与 `hardware.yaml` 的关系：V1.5 让 `_detect_vllm_gpus*` 退化为读 `hardware.yaml` 中 `role: llm` group 的 `gpus`，不再自行探测，避免两套来源打架。

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

### 3.3 主进程 ↔ image/TTS Runner 协议

走 `multiprocessing.Pipe` + msgpack 编码（dev 模式可 `NOUS_IPC_FORMAT=json` fallback 便于 journalctl 调试）。**仅 image/TTS runner 走此协议**；LLM runner 不收 RunNode，主进程直连其 vLLM HTTP 端口。

**实现约束（F1）**：`multiprocessing.Pipe` 对象不可直接 `await`。
- pipe-reader 侧：用 `loop.connect_read_pipe` / `loop.add_reader` 把 fd 注册进 event loop，或起一个读线程桥到 `asyncio.Queue`。
- pipe-writer 侧：`Pipe.send` 无 timeout 参数。§4.1「IPC pipe 阻塞 5s 超时」需要一个写线程 + `Queue.join` 超时，或把 fd 设非阻塞后自行轮询。不能假设 `send` 会及时返回。

```python
# 主进程 → image/TTS runner
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
    node_type: str           # 仅 image / tts
    model_key: str | None
    inputs: dict             # 大 tensor → host-pinned 引用；小标量/dict → 内联 msgpack
    is_deterministic: bool   # V1.5 仅落表，V1.6 缓存用

class Abort:
    kind: Literal["abort"]
    task_id: int
    node_id: str | None

class Ping:
    kind: Literal["ping"]

# image/TTS runner → 主进程
class NodeResult:
    kind: Literal["node_result"]
    task_id: int
    node_id: str
    status: Literal["completed", "failed", "cancelled"]
    outputs: dict | None     # image: {path: "outputs/{task_id}/..", meta: {...}}
    error: str | None
    duration_ms: int

class NodeProgress:
    kind: Literal["node_progress"]
    task_id: int
    node_id: str
    progress: float          # 0.0 ~ 1.0
    detail: str | None       # "step 12/30"

class ModelEvent:
    kind: Literal["model_event"]
    event: Literal["loaded", "unloaded", "load_failed"]
    model_key: str
    error: str | None

class Pong:
    kind: Literal["pong"]
    runner_id: str
    loaded_models: list[str]
```

无 `IPCRef`。无 vLLM 相关消息（LLM runner 不走此协议）。

### 3.4 跨 runner 张量传输（无 CUDA IPC）

CUDA IPC / `from_ipc_handle` / `IPCRef` **已删除**。理由：每 group 恰好 1 runner → 跨 runner 永远跨 group → P2P 路径在 V1.5 拓扑下是死代码。

跨 runner 中间 tensor 一律走 host-pinned memory 中转（见 §2.3）。实现：
- runner 内 D→H 拷到 `torch.empty(..., pin_memory=True)` buffer
- buffer 经 shared memory 段传给主进程 / 目标 runner
- 目标 runner H→D 拷回 GPU

小标量、dict、文件路径不过 host-pinned，直接内联进 msgpack pipe 消息。

### 3.5 主进程内部数据结构

复用 model_manager 已有的 `_load_failures` dict 与 `evict_lru(gpu_index)`，不重新发明。

```python
@dataclass
class GroupScheduler:
    group_id: str
    queue: asyncio.PriorityQueue[QueuedTask]
    runner_client: RunnerClient | None        # LLM group 为 None（无 dispatch 队列）
    cancel_events: dict[int, asyncio.Event]
    inflight_tasks: dict[int, asyncio.Task]

@dataclass(order=True)
class QueuedTask:
    sort_key: tuple[int, datetime]     # (priority, queued_at)
    task_id: int = field(compare=False)
    workflow_spec: dict = field(compare=False)

@dataclass
class TaskSnapshot:
    task_id: int
    status: str
    # ... 其余字段省略 ...
    db_synced: bool                    # DB 写成功 = True；降级期写失败 = False

class TaskRingBuffer:
    """最近 200 条 task 快照，O(1) 读，给 TaskPanel / Dashboard 用。"""
    _items: collections.deque[TaskSnapshot]   # maxlen=200
    _by_id: dict[int, TaskSnapshot]
```

DB 是真相源（survives restart），ring buffer 是热缓存（survives request burst）。`db_synced` 标志驱动 §4.6 的 reconcile。

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

设计原则：**故障域局限到 GPU Runner 子进程**。主进程永远不应因为某 runner crash 或某模型 OOM 而挂掉。

### 4.1 故障矩阵

| 故障类型 | 影响域 | 检测手段 | 恢复策略 |
|---|---|---|---|
| image/TTS Runner crash | 该 GPU group 全部任务 | pipe EOF / ping 超时 | 主进程重启 runner + resident 重新 preload |
| LLM Runner crash | 该 LLM 全部推理 | ping 超时 / vLLM HTTP 失联 | 重启 runner → re-spawn vLLM + preload |
| 模型 load OOM | 该模型 | adapter.load() 抛 CUDA OOM | LRU evict 同 group + 重试一次；二次失败标 load_failed |
| 模型 load 文件丢失/损坏 | 该模型 | adapter.load() 抛 FileNotFound | 标 `_load_failures[model_key]`，写 health，不重试 |
| vLLM 子进程启动失败 | 该 LLM | scanner / health check 失败 | kill orphan，标 load_failed |
| 节点执行 OOM | 该节点 | adapter.run() 抛 CUDA OOM | 节点 fail；不重试；workflow 整体 fail |
| 节点超时 | 该节点 | `asyncio.wait_for` 超时 | set cancel flag（穿过 to_thread）+ abort；节点 fail；workflow fail |
| 节点 cancel | 该节点 + 后续 | cancel_event 触发 | 节点 cancelled；workflow status=cancelled |
| IPC pipe 阻塞 | 该 runner 当前 task | 写 pipe 5s 超时（写线程/非阻塞 fd，见 §3.3） | 视为 runner 假死 → SIGKILL + 重启 |
| DB 不可达 | 所有 task 持久化 | sqlalchemy 异常 | 降级：继续执行 + ring buffer（`db_synced=false`）+ WS；DB 恢复后 reconcile |
| WebSocket 推送失败 | 仅前端实时性 | send 异常 | 静默吞掉，下次心跳重连 |
| Runner pipe 拥塞 | 该 runner 响应延迟 | queue size > 阈值 | 反压：scheduler 暂停向该 runner 派新 task |

### 4.2 Runner 生命周期

```
[main process startup]
  ├─ 读 hardware.yaml → groups[] → 决定要起几个 runner
  └─ for each group:
       fork runner subprocess
       wait runner 发 "ready" 消息（含 runner_id + GPU list）
       ▼
       [resident preload]
         按 preload_order 升序遍历 group_hint 匹配的 resident:true 模型
         image/TTS: send LoadModel → 等 ModelEvent(loaded/load_failed)
         LLM:       runner spawn vLLM → 等 health 通过
         load_failed 写 _load_failures[model_key] = error
         不阻断后续 preload，不阻断 API server start
```

#### Crash 检测 + 重启

```python
class RunnerSupervisor:
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
        # 1. 终结 runner（SIGTERM 5s → SIGKILL）
        # 2. 当前所有 inflight task 标 failed (runner_crashed)
        # 3. backoff (防 crash 风暴)
        # 4. GPU-free gate：轮询 nvidia-smi 直到该 group 的 GPU 显存回落到基线
        #    （F2：OOM/native fault 后 CUDA context 回收是异步的，backoff 可能太短）
        # 5. 重新 fork + preload resident
        # 6. 成功跑 30 min 后 reset crash_count
```

**GPU-free gate（F2）**：re-preload 前必须确认 `nvidia-smi` 显示该 group 的 GPU 显存已回落——死进程的 CUDA context 回收异步，纯 backoff `[5,15,60,300]` 不保证 context 已清。gate 失败则继续等并延长 backoff。

重启期间：该 group 的 PriorityQueue 不动，新 task 继续入队但 dispatcher 暂停。runner 起来后继续派发。

**Inflight task 处理策略**：全部标 failed (runner_crashed)，不重试。理由：不知道副作用是否已写（文件、token 计费），重试可能造成重复副作用。

### 4.3 OOM 处理

复用 model_manager 已有 `_load_failures` + `evict_lru`：

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
                    self.evict_lru(self._gpu_index_for(model_key))
                    continue
                self._load_failures[model_key] = f"OOM after evict: {e}"
                raise
            except Exception as e:
                self._load_failures[model_key] = str(e)
                raise
```

二次 OOM 后**用户手动 retry**：Dashboard / runner 泳道显示加载失败态 + Retry 按钮触发重新加载。不做自动指数退避（避免用户调试时被反复 OOM 干扰）。

### 4.4 Runner 内部并发 + Cancel 双层

#### Runner 内部并发（D9 / C3，仅 image/TTS runner）

image/TTS runner 内跑**两个 asyncio task**：

```
┌─────────────────── image/TTS Runner ───────────────────┐
│                                                        │
│  ┌─ pipe-reader task ─┐      ┌─ node-executor task ─┐  │
│  │ 持续读 pipe 消息   │      │ 从队列取 RunNode      │  │
│  │ RunNode → 入队     │      │ adapter.run(...)      │  │
│  │ Abort  → set       │─────▶│ 长循环轮询 CancelToken│  │
│  │   threading.Event  │ flag │ / callback 检查 Event │  │
│  └────────────────────┘      └───────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

- pipe-reader 永远不阻塞在 adapter.run 上，收到 Abort 能立即置位
- node-executor 跑 adapter.run；`adapter.run` 须为 **async 可让出**，否则 pipe-reader 收不到调度
- 跨线程信号用 `threading.Event`（adapter 的扩散循环在 `to_thread` 里跑，见下）

#### Cancel 双层

```python
# 节点边界（主进程 WorkflowExecutor）
async def execute_node(self, node, cancel_event):
    if cancel_event.is_set():
        raise NodeCancelled()
    if is_dispatch_node(node):
        return await self._dispatch_to_runner(node, cancel_event)
    return await self._inline_vllm_http(node, cancel_event)   # llm 节点

# within-node（image/TTS runner 内 adapter）—— D14：含 adapter 重写
# 现状：image_diffusers.py 把扩散 pipe 当 to_thread 里一个不透明阻塞调用，
#       wait_for 取消 to_thread 不会停 CUDA kernel。V1.5 重写 adapter：
class CancelFlag:
    """pipe-reader 收到 Abort 或 wait_for 超时都 set 同一个 Event。"""
    event: threading.Event

def _on_step_end(pipe, step, timestep, kwargs):
    # diffusers callback_on_step_end，在 to_thread 工作线程里跑
    if cancel_flag.event.is_set():
        raise NodeCancelled()       # 中断扩散循环，停 CUDA kernel
    return kwargs

# image adapter：接 diffusers callback_on_step_end
pipe(prompt, num_inference_steps=30, callback_on_step_end=_on_step_end)

# 超时路径走同一 flag
try:
    result = await asyncio.wait_for(asyncio.to_thread(run_pipe), timeout=T)
except asyncio.TimeoutError:
    cancel_flag.event.set()         # 让 callback 在下一 step 抛 NodeCancelled
    raise NodeTimeout()

# LLM streaming（主进程 executor / compat 路由，直连 vLLM HTTP）
async for token in vllm_http_stream:
    if cancel_event.is_set():
        await vllm_http_abort(request_id)
        raise NodeCancelled()
    yield token
```

TTS / VAE / 短节点只在边界 check（计算太短，within-node check overhead 反而大）。

**关键性质（D14）**：`callback_on_step_end` 是唯一能让 cancel / timeout 信号穿过 `to_thread` 边界、真正停掉 CUDA kernel 的机制。`asyncio.wait_for` 单独用不行——它取消的是 awaiting，`to_thread` 里的 CUDA kernel 照跑。

### 4.5 隔离边界

```
┌────────────────────────────────────────────────────────┐
│ Main Process                                           │
│  - Event Loop A                                        │
│  - DB session pool                                     │
│  - RunnerSupervisor × N                                │
│  - GroupScheduler × N (image/TTS group 有 PriorityQueue)│
│  - WorkflowExecutor（llm 节点 inline HTTP 调 vLLM）     │
│  - cancel_events: dict[task_id, asyncio.Event]         │
└────────────────────────────────────────────────────────┘
       │ multiprocessing.Pipe (msgpack)  │ HTTP
       │ image/TTS runner                │ 直连 vLLM
       ▼                                 ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ image/TTS Runner         │   │ LLM Runner               │
│  - Event Loop B (独立)    │   │  - 只管 vLLM 子进程       │
│  - ModelManager (单例)    │   │    生命周期               │
│    - per-model asyncio.Lock│  │  - 不串行化推理请求       │
│    - LRU evict 限本 group │   │  - vLLM 自身又是 subprocess│
│  - pipe-reader + executor │   │    crash 由 runner 重启它  │
└──────────────────────────┘   └──────────────────────────┘
```

关键性质：
- 主进程 DB session / HTTP handler / WebSocket 跑 Event Loop A，runner crash 不影响
- image/TTS runner 内某 model OOM 不影响同 runner 其他 model（per-model lock + GPU 内存隔离）
- runner 内某 model native segfault 拖死整个 runner → 走 §4.2 restart
- LLM runner 与 vLLM 是两级子进程：vLLM crash → LLM runner 重启它；LLM runner crash → 主进程重启它

#### Inline 执行点改道清单（D6）

V1 在主进程内 inline 执行推理的调用点，V1.5 全部改道：

| 调用点 | V1 行为 | V1.5 改道 |
|---|---|---|
| `openai_compat` | 主进程内调 adapter | 直连 vLLM HTTP（LLM runner 已 spawn）|
| `anthropic_compat` | 同上 | 直连 vLLM HTTP |
| `ollama_compat` | 同上 | 直连 vLLM HTTP |
| `responses.py` | 同上 | 直连 vLLM HTTP |
| `workflow_executor` 的 llm 节点 | 同上 | executor inline HTTP 调 vLLM |
| `workflow_executor` 的 image/TTS 节点 | 主进程内调 adapter | dispatch 到对应 runner 串行队列 |

compat 路由本来就是同步契约，改道后依然同步、不进队列、零 per-token pipe 开销（D8）。

### 4.6 DB 不可达降级 + reconcile（D7）

DB 写入失败仅记 log，业务继续执行，靠 ring buffer + WebSocket 推送给前端。

**reconcile 机制**：
- 每条 `TaskSnapshot` 有 `db_synced: bool`。DB 写成功 → `True`；降级期写失败 → `False`。
- 后台任务周期性探测 DB 可达性。DB 恢复后遍历 ring buffer，把所有 `db_synced=false` 的快照批量补写，成功后置 `True`。
- 代价：DB 故障期间，ring buffer 容量（200）外被 evict 的历史无法补写——这部分丢失。

理由：用户场景是单 admin 推理 infra，DB 短暂不可达不应中断推理流。本机 Postgres + systemd 同机，DB 独立不可达极罕见。

### 4.7 关键边界 case

1. **Cancel 已 dispatch 但 runner 还没收到**：主进程标 task=cancelled，等 NodeResult 回来直接丢弃
2. **多个 task 同时 cancel**：cancel_events 是 dict[task_id]，互不影响
3. **API server 重启**：启动时扫 DB：
   - status=running → 标 failed (server_restarted)
   - status=queued → 重新入对应 group queue
4. **某 group 队列堆积 >1000**：拒绝新入队，返回 503 + Retry-After
5. **罕见双重故障窗口（G3，明确接受）**：§2.2「DB enqueue 前 commit」依赖 DB 可达。若**「DB 在 enqueue 时就不可达」+「主进程在 DB 恢复前重启」**同时发生，该 task 既没进 DB、ring buffer 也随重启清空——**该 task 丢失**。V1.5 **不建 durable spool**：本机 Postgres + systemd 同机，DB 独立不可达本就极罕见，叠加主进程重启窗口概率可忽略，为它建持久化层是 over-engineering。

---

## 5. 测试策略

### 5.1 测试分层

```
┌─────────────────────┐
│ E2E (真 GPU)         │  ~10 个，dev box 跑，CI skip
│ pytest -m e2e        │
├─────────────────────┤
│ Integration (mock    │  ~30 个，CI 默认跑
│   runner subprocess) │
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
| `RunnerSupervisor.RESTART_BACKOFF` | crash_count 累加 + reset；exponential backoff 序列；GPU-free gate |
| `TaskRingBuffer` | maxlen=200 evict 最旧、by_id lookup、list_recent limit、`db_synced` 标志 |
| `hardware.yaml` 解析 | 2gpu / 3gpu 两份 yaml；按 groups[] 决定 runner 数 |
| `preload_order` 排序 | resident:true + preload_order 升序；null 排最后 |
| NodeSpec 校验 | tp>1 强制 nvlink:true；device 字段格式 |
| msgpack 序列化 | 编码/解码往返；dev 模式 JSON fallback；大 tensor 走 host-pinned 分支 |
| Cancel 双层 check | 边界路径 + within-node CancelFlag |
| 节点分流 | dispatch 节点 vs inline-HTTP 节点判定 |

### 5.3 Integration 测试（mock runner subprocess）

主进程 + fake runner（纯 Python，不加载真模型），跑通完整 IPC 协议。

| 场景 | 验证 |
|---|---|
| Workflow 完整生命周期 | enqueue → dispatch → run_node → completed；DB + ring buffer + WS |
| 优先级抢占 | batch task A 先入，interactive B 后入，B 先 dispatch |
| Cancel inflight | dispatch 后 cancel → fake runner 模拟 Abort → status=cancelled |
| Runner crash 检测 | fake runner kill self → PING_TIMEOUT 内检测 → inflight 标 failed |
| Runner 重启 resident preload | 重启后收 LoadModel 序列；preload_order 升序 |
| 模型 load_failed 不阻断 | 某 model_key 回 load_failed → 后续继续；/health 暴露 failure |
| 队列堆积 503 | 灌 1001 个 task → 第 1001 个 503 + Retry-After |
| API server 重启恢复 | DB 中 status=running → failed (server_restarted)；queued → 重新入队 |
| **[回归] Lane 0 monitor** | Lane 0 后 monitor.py / gpu_monitor.py 仍正确报告加载状态 + idle-TTL 卸载仍生效 |
| **[回归] compat 路由** | A5 改道后 4 个 compat 路由（openai/anthropic/ollama/responses）仍产出正确输出 |
| **[回归] workflow 异步契约** | workflow inline → queued（D17 纯异步 202）后执行结果不变 |
| **[回归] ModelManager 合并** | src/gpu/model_manager.py 合并/删除后所有调用点仍工作 |
| LLM runner 不串行化 | LLM lifecycle runner 不串行化 vLLM 请求（并发请求同时在飞）|
| Runner 内部并发 | pipe-reader + executor task 分离；Abort-during-node-execution within-node 生效 |
| 跨 runner tensor | host-pinned 跨 runner tensor 传递正确还原 |
| DB reconcile | DB 恢复后 `db_synced=false` 条目批量补写 |
| 混合节点 workflow | image dispatch + llm inline HTTP；两路并发；结果汇合正确 |

CI 跑 `pytest -m integration`，5 分钟内完成。

### 5.4 E2E 测试（真 GPU，dev box）

每次 PR 合并前手动跑。

| 场景 | 验证 |
|---|---|
| 多 group 并发 | 真双卡 utilization 同时拉满（Pro 6000 到货后 image + LLM 并发）|
| vLLM tp=2 | Qwen3-35B 在 GPU 0+1 NVLink pair 上 load；nvidia-smi 看 P2P 流量 |
| **[D14] image within-node cancel** | 30 step sampler，第 5 步 cancel → `callback_on_step_end` 接入 + cancel/timeout 信号穿过 to_thread（threading.Event）→ <500ms 停 CUDA kernel |
| OOM evict 路径 | 故意 load 超 VRAM → LRU evict + 重试；二次 OOM → load_failed |
| Runner 真 crash 恢复 | `kill -9` runner → GPU-free gate → 重启 + resident 重 load |
| 主进程重启幂等 | 已有 vLLM orphan → reconnect healthy / kill unhealthy（回归） |

### 5.5 故障注入（`tests/chaos/`，每周手动跑）

```python
# test_runner_crash_storm.py
async def test_runner_repeated_crashes():
    """连续 5 次 runner crash → 验证 backoff + GPU-free gate + 主进程不挂"""

# test_pipe_slow_consumer.py
async def test_runner_blocks_pipe():
    """fake runner 不读 pipe → 主进程 5s 写超时（写线程/非阻塞 fd）+ 反压"""

# test_db_flaky.py
async def test_db_intermittent_failures():
    """50% 概率 DB OperationalError，soak 1000 task → ring buffer + reconcile 一致"""
```

### 5.6 测试基础设施

| 已有 | 复用 |
|---|---|
| `tests/conftest.py` 强制 `ADMIN_PASSWORD=""` | 复用 |
| `NOUS_DISABLE_FRONTEND_MOUNT=1` | 复用 |
| `pytest-asyncio` | 复用 |

| 新增 | 用途 |
|---|---|
| `tests/fixtures/fake_runner.py` | mock image/TTS runner subprocess，可配置 crash/slow/fail-load |
| `tests/fixtures/fake_vllm.py` | mock vLLM HTTP 端点，给 LLM 直连路径用 |
| `tests/fixtures/hardware_topo.py` | 2gpu / 3gpu hardware.yaml fixture |
| `pytest.ini` markers | `e2e`, `integration`, `chaos` |

---

## 6. 前端设计（TaskPanel 重构 + Dashboard 标识）

plan-design-review 7-pass 决策。V1.5 前端从「3-tab 扁平列表」**重构**为 Buildkite 风 per-runner 泳道——这是 TaskPanel 重构，超原「扩展」scope。设计系统复用 `TaskPanel.tsx` 既有 CSS vars（`--bg/--bg-accent/--border/--text/--muted/--accent/--accent-2/--info/--mono`）+ lucide 图标 + status badge 模式。无正式 DESIGN.md（建议某天单独跑 /design-consultation，非 V1.5 阻塞）。

### 6.1 TaskPanel 布局（Buildkite 风）

```
┌─────────────── TaskPanel ───────────────┐
│  per-runner 泳道区（视觉 hero）           │
│  ┌────────────────────────────────────┐ │
│  │ ● Runner-I (image)   busy          │ │
│  │   flux2-人物立绘  ▓▓▓▓▓░░ step18/30│ │
│  │   排队 3  ▸                         │ │  ← 可点击展开
│  ├────────────────────────────────────┤ │
│  │ ● Runner-L (llm)     idle          │ │
│  │   排队 0                            │ │
│  ├────────────────────────────────────┤ │
│  │ ◐ Runner-T (tts)     重启中 2/4    │ │  ← 异常态内联
│  └────────────────────────────────────┘ │
│  ─────────────────────────────────────── │
│  最近完成                                │
│  ┌────────────────────────────────────┐ │
│  │ [缩略图] flux2-人物立绘  34s  done  │ │  ← image 显示缩略图
│  │ [缩略图] sd-背景        12s  done  │ │
│  │ [audio]  cosy-旁白      8s  done   │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

- **顶部 per-runner 泳道区（DD3）** — 每条泳道 = 一个 GPU runner 的当前任务 + 进度条 + 排队数。取代原「3-tab + 扁平列表 + 独立队列折叠区」。
- **下方「最近完成」列表（DD3）** — 替代原扁平 history 列表。
- **image 缩略图历史（DD9）** — image 类任务完成后，「最近完成」直接显示输出缩略图（数据源 = `outputs/{task_id}/`），不只是文字 + ImageIcon。并入 Lane I。

### 6.2 runner 泳道异常态（DD5）

泳道内联表达，状态文字始终伴随色点（色盲 a11y）：

| 态 | 表达 |
|---|---|
| idle | 灰点 + 「idle」 |
| busy | 绿点 + 当前任务名 + 进度条 |
| 重启中 | 黄色脉冲点 + 「重启中 2/4」（backoff 第几次）|
| 加载失败 | 红点 + 「加载失败: qwen3-35b OOM」+ **Retry** 按钮 |

### 6.3 Run 反馈 + 完成通知

- **点 Run 反馈（DD4）** — D17 异步后，点 Run → toast「任务已入队 · image runner」+ IconRail 任务图标 badge 计数；**面板不自动打开**；toast 带「查看」跳转。
- **任务完成通知（DD6）** — 完成 → app 内 toast「flux2-人物立绘 完成 · 34s」（带「查看」）+ badge 更新 + 浏览器通知（Notification API）。失败同样发。浏览器通知：权限首次询问，拒绝则降级为仅 toast；仅在页面失焦时发系统通知。

### 6.4 排队位置可见（DD8）

runner 泳道的「排队 N」可点击展开成有序列表，每条带序号 #1 #2 #3，刚提交的 / 当前用户的任务高亮。

### 6.5 响应式（DD7）

TaskPanel / Dashboard 做响应式：<768px 抽屉变全屏布局，runner 泳道堆叠。

### 6.6 a11y 必加清单（Pass 6，无备选）

- 折叠 toggle（排队展开、泳道展开）用真 `<button>` + `aria-expanded`
- runner 状态文字始终伴随色点（色盲不能只靠颜色）
- toast 用 `aria-live`
- Retry / 取消按钮键盘可达
- 浏览器通知权限流：首次询问 → 拒绝降级 toast-only

---

## V1.5 vs V1.6 节点输出缓存取舍

ComfyUI 的最大优势之一是**节点输出缓存**：按节点输入 hash 缓存输出，重跑同 workflow 只改 seed 时前面的 CLIP encode、VAE decode 等节点全 hit cache，几秒内完成。但 V1.5 暂不实现，理由如下。

### 为什么 V1.5 不做

1. **架构基础不稳就上层优化**会埋雷：当前 inference race / 调度 / OOM 三件事还没解决，缓存只是性能加速。
2. **正确性门槛高**：缓存命中 = 必须保证"相同输入 + 相同 model 版本 + 相同节点代码"产出相同输出。涉及输入 hash 标准化、`is_deterministic` 强制声明、model 版本指纹。做不对会出"明明改了参数还看到旧结果"的脏 bug。
3. **存储语义未定**：缓存放 runner 进程内存？跨 runner 共享？容量怎么淘汰？V1.5 的 runner 拓扑刚刚确定，先观察使用模式再决定。

### V1.5 已经做的前置准备

- NodeSpec 加 `is_deterministic: bool` 字段（仅落 ExecutionTask.node_timings，不影响行为）
- WSEvent 词表预留 `execution_cached` 类型
- `node_timings` JSON 字段为每节点保留 `cached: bool`（V1.5 永远 false）

### V1.6 实现条件

实现节点输出缓存前必须先解决：

1. **输入 hash 标准化**：每种 NodeInput 类型必须有 deterministic hash 方法（包括 tensor、dict、file path）
2. **`is_deterministic` 强制声明**：所有 NodeSpec 必须显式声明；缺失视为 false（不缓存）
3. **Model 版本指纹**：models.yaml 加 `version_hash`（基于 weight file mtime + size），缓存 key 包含此字段
4. **存储拓扑决策**：单 runner 内内存 LRU（最简单）vs 跨 runner 共享磁盘缓存

V1.6 设计将单独写 spec doc。

---

## 实施分 Lane

按依赖顺序拆 PR。Lane 0 是前置的零行为变化 refactor，先于所有 V1.5 Lane。

| Lane | 内容 | 依赖 |
|---|---|---|
| **0** | **调度器整合（前置，零行为变化）**：删 `model_scheduler.py`，`monitor.py` + `gpu_monitor.py` 改用 `model_manager`；删除前把 `get_llm_base_url()`（`model_scheduler.py:233`）重新安置到合并后的 model_manager（G4）；**设计并实现** `services/model_manager.py`（asyncio.Lock）+ `src/gpu/model_manager.py`（threading.Lock + VRAMTracker）的合并——后者的 `VRAMTracker` / `can_load` 是 NVLink allocator 需要的真实 VRAM 核算，不是重叠是互补；合并后的类要能在跑自己 event loop 的 runner 子进程内工作（G5） | 无 |
| A | `hardware.yaml`（2gpu/3gpu 两份，manual-only）+ GPUAllocator 重构（NVLink-aware，按 yaml groups[] 动态决定 runner 数） | 0 |
| B | `execution_tasks` schema migration + TaskRingBuffer（含 `db_synced` 标志） | 无 |
| C | RunnerSupervisor + image/TTS runner 子进程框架（fake adapter，跑通 IPC；pipe-reader + executor 双 task；F1 pipe 不可 await 的实现约束） | A |
| D | ModelManager 迁入 image/TTS runner（image runner 先迁，验证 per-model lock） | B, C |
| E | LLM Runner（只管 vLLM 生命周期：spawn/health/preload/abort/OOM-restart）+ 主进程 compat 路由 / executor 直连 vLLM HTTP（D6/D8 改道清单） | D, S |
| F | TTS runner 迁入 | D |
| G | GroupScheduler + priority + cancel 双层 + **adapter 重写**（image adapter 接 diffusers `callback_on_step_end`，cancel/timeout 走同一 `threading.Event` 穿过 to_thread，D14） | C |
| H | resident `preload_order` + `_load_failures` + `/health` 扩展 + Runner 重启 GPU-free gate（F2） | D |
| **S** | **workflow_executor 重写 + `/run` 契约变更**：`workflow_executor.py` 的 `execute()`（现 `:102` 扁平顺序循环）拆成 dispatch 节点 vs inline-HTTP 节点（D10）；`execute_workflow_direct`（`workflows.py:142`）的 inline 执行收编；`/v1/workflows/{id}/run` 改纯异步 202 + task_id（D17）；mediahub 等上游迁移指引 | C |
| I | TaskPanel 重构为 Buildkite 风 runner 泳道（DD3-DD8）+ image 缩略图历史（DD9）+ 响应式 + a11y + Dashboard runner 标识 | B, G, S |
| J | Integration + chaos 测试套（含 §5 的 4 项 CRITICAL 回归） | 0、A 到 I 全部 |

每 Lane 一个 PR，本地 ruff + tsc + vite build 过 → push → CI 绿 → merge。

### `/run` 契约变更 + 上游迁移指引（D17）

`/v1/workflows/{id}/run`：

| | V1 | V1.5 |
|---|---|---|
| 行为 | 同步执行，阻塞到完成才返回 result | 入队 → 立即返回 `202 + {task_id}` |
| 客户端拿结果 | 响应 body | 轮询 `/v1/tasks/{id}` 或订阅 WS `/ws/workflow/{id}` |

**注意**：单次 LLM 调用的同步体验**不变**——compat 路由（openai/anthropic/ollama/responses）按 D6/D8 直连 vLLM HTTP，本来就同步、不进队列。D17 只改多节点 workflow 端点。

**mediahub 等上游迁移**：上游调 `/run` 的代码改成「拿 task_id → 轮询或订阅」。迁移期可保留一个 `/run?wait=true` 兼容 flag（服务端代为轮询后返回），但标记 deprecated。

---

## Open Questions

无。本 spec 决策点全部锁定（plan-eng-review D1-D18 + plan-design-review DD3-DD9 全部并入正文）。

---

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
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | DECISIONS_INCORPORATED | 13 issues (7 arch + 1 quality-decisions + 1 perf + 4 critical regressions), 2 critical gaps; 18 decisions confirmed AND folded into spec body |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | DECISIONS_INCORPORATED | score 2/10 → 8/10, 7 decisions (DD3–DD9) folded into §6; thumbnail history pulled into V1.5 Lane I |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE VOICE:** codex 配置损坏（`tui.alternate_screen`），回退 Claude 独立子代理。翻出 G1/G2（adapter 非 async-cancellable）、G3（DB 恢复矛盾）、G4（`get_llm_base_url` 接缝）、G5（两个 ModelManager 互补非重叠）、F1/F2/S3，战略挑战 D1。
- **CROSS-MODEL:** 4 处 tension（D1 scope / within-node cancel / topo 探测 / DB reconcile）全部经 AskUserQuestion 由用户裁决（D13–D17）。
- **DESIGN:** plan-design-review 7-pass 完成。TaskPanel 从「3-tab 扁平列表」重构为 Buildkite 风 runner 泳道（DD3，见 §6）；点 Run 反馈、runner 异常态、完成通知、响应式、排队位置全部并入 §6。
- **INCORPORATED:** plan-eng-review 18 项 + plan-design-review 7 项决策全部并入 spec 正文；transient review-tracking 节已删除。
- **CRITICAL GAPS（已设计落地，标记为实现风险）:**
  - **G2 — to_thread cancel 泄漏**：`asyncio.wait_for` 取消 `to_thread` 不停 CUDA kernel。设计方案见 §4.4：image adapter 接 diffusers `callback_on_step_end` + `threading.Event`，cancel 与 timeout 走同一 flag。**实现风险**：依赖每个 image adapter 都正确接入 callback，遗漏一个就退化为不可 cancel；Lane G 必须逐 adapter 验证。
  - **F1 — Pipe 无 timeout**：`multiprocessing.Pipe.send` 无 timeout 参数，`Pipe` 对象不可 await。设计方案见 §3.3：pipe-reader 用 `connect_read_pipe`/`add_reader`/线程桥，pipe-writer 用写线程或非阻塞 fd 实现 5s 超时。**实现风险**：写线程 + 超时检测的正确性边界微妙（部分写、fd 状态），Lane C 需 chaos 测试 `test_pipe_slow_consumer` 专门压。
- **VERDICT:** MOVING TO CLEARED — 25 项决策已全部并入 spec 正文，2 个 CRITICAL GAP（G2 to_thread cancel、F1 Pipe 无 timeout）已在 §4.4 / §3.3 设计落地并标记为实现风险，由 Lane G / Lane C 承接验证。spec body 与决策一致，可转 `superpowers:writing-plans`。
