# V1.5 Lane B: execution_tasks schema 扩展 + TaskRingBuffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `execution_tasks` 表加上 V1.5 调度所需的 8 个新列（priority / gpu_group / runner_id / queued_at / started_at / finished_at / node_timings / cancel_reason），并实现一个纯内存的 `TaskRingBuffer`（200 条最近 task 快照，每条带 `db_synced` 标志）。这是数据层 Lane —— **不**把 ring buffer 接进 scheduler（那是后续 Lane G/I 的事）。

**Architecture:** 两块互不依赖的交付物：
1. **Schema 扩展** —— `ExecutionTask` ORM 模型加 8 个 nullable 列 + 一份手写 SQL migration。新列全部 nullable，旧行保持 NULL，`ADD COLUMN IF NOT EXISTS` 幂等。
2. **TaskRingBuffer** —— `src/services/task_ring_buffer.py` 新模块：`TaskSnapshot` dataclass + `TaskRingBuffer` 类。内部 `collections.deque(maxlen=200)` + `dict[int, TaskSnapshot]` 双索引，O(1) push / by-id 读 / list_recent。`db_synced: bool` 字段为 §4.6 的 DB-recovery reconcile 留钩子（本 Lane 只存标志、提供翻转 API，不实现 reconcile loop）。

**Tech Stack:** Python 3.12 / SQLAlchemy 2.x（`Mapped` / `mapped_column`）/ pytest（`asyncio_mode = "auto"`）/ 标准库 `collections.deque` + `dataclasses`。无第三方依赖。

> **注意 — 与 task 简报的偏差（已核实，须知会）：本仓库没有 alembic。** 简报要求「写一个 alembic migration」，但 `backend/` 下无 `alembic/`、无 `alembic.ini`、无 `alembic/versions/`。本仓库的 schema 演进实际是两条腿：
> - **建表**：`src/api/main.py` lifespan 里 `Base.metadata.create_all`（新库 / 新表自动建）。
> - **增量改 schema**：手写 `backend/migrations/*.sql` 文件，部署时 `psql $DATABASE_URL -f ...` 手动跑（见 `migrations/2026-04-23-keys-m10.sql`、`migrations/2026-04-25-passkey-totp.sql`，均 `BEGIN; ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...; COMMIT;` 单事务幂等）。
>
> 所以本 Lane 的「migration」= 一份手写 SQL 文件 `backend/migrations/2026-05-14-v15-execution-tasks.sql`，对齐既有风格。**不引入 alembic** —— 为单 admin 推理 infra 一个 8 列的 ADD COLUMN 引入整套迁移框架是 over-engineering，且与既有 4 份 migration 的运维方式不一致。下游 Lane（A/C/...）若需要 alembic 是另一个独立决策，不在 Lane B scope。

---

## File Structure

| 文件 | Lane B 动作 | 责任 |
|---|---|---|
| `backend/src/models/execution_task.py` | **修改** | `ExecutionTask` ORM 加 8 个 V1.5 nullable 列 |
| `backend/migrations/2026-05-14-v15-execution-tasks.sql` | **新建** | 手写 SQL：`ALTER TABLE execution_tasks ADD COLUMN IF NOT EXISTS ...` ×8，单事务幂等 |
| `backend/src/services/task_ring_buffer.py` | **新建** | `TaskSnapshot` dataclass + `TaskRingBuffer` 类（deque(maxlen=200) + by-id dict + db_synced） |
| `backend/tests/test_execution_task_schema.py` | **新建** | ORM 新列可写可读、默认值、nullable 回归 |
| `backend/tests/test_task_ring_buffer.py` | **新建** | maxlen evict、by-id lookup、list_recent limit、db_synced 翻转、update-in-place |

---

## Task 1: 给 `ExecutionTask` ORM 加 V1.5 新列

spec §3.1 明确列出 V1.5 新增的 8 列。先在 ORM 层加上（`db_session` / `db_client` fixture 跑的是 SQLite + `Base.metadata.create_all`，新列即时生效，不依赖 SQL migration）。

**Files:**
- Modify: `backend/src/models/execution_task.py`
- Test: `backend/tests/test_execution_task_schema.py`（新建）

- [ ] **Step 1: 跑现有 task 相关 suite 建基线**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_api_execution_tasks.py tests/test_api_tasks.py -q`
Expected: PASS（记下通过数，作为改 ORM 后的回归对照）。

- [ ] **Step 2: 写失败测试 —— 新列可写可读 + 默认值 + nullable**

新建 `backend/tests/test_execution_task_schema.py`：
```python
"""Lane B: ExecutionTask V1.5 新列 schema 回归。

新列全部 nullable（priority 例外：有 default=10），旧调用方不传也能 insert。
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from src.models.execution_task import ExecutionTask

pytestmark = pytest.mark.anyio


async def test_v15_columns_default_to_null(db_session):
    """不传 V1.5 列时，insert 成功；priority 落 default=10，其余落 NULL。"""
    task = ExecutionTask(workflow_name="laneB-defaults", status="queued")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    assert task.priority == 10  # default
    assert task.gpu_group is None
    assert task.runner_id is None
    assert task.queued_at is None
    assert task.started_at is None
    assert task.finished_at is None
    assert task.node_timings is None
    assert task.cancel_reason is None


async def test_v15_columns_round_trip(db_session):
    """写入全部 V1.5 列后能原样读回。"""
    now = datetime(2026, 5, 14, 10, 0, 0, tzinfo=timezone.utc)
    task = ExecutionTask(
        workflow_name="laneB-roundtrip",
        status="completed",
        priority=0,
        gpu_group="llm-tp",
        runner_id="runner-i",
        queued_at=now,
        started_at=now,
        finished_at=now,
        node_timings={"node_a": {"duration_ms": 1200, "cached": False}},
        cancel_reason=None,
    )
    db_session.add(task)
    await db_session.commit()
    task_id = task.id

    db_session.expire_all()
    fetched = (
        await db_session.execute(
            select(ExecutionTask).where(ExecutionTask.id == task_id)
        )
    ).scalar_one()

    assert fetched.priority == 0
    assert fetched.gpu_group == "llm-tp"
    assert fetched.runner_id == "runner-i"
    assert fetched.queued_at == now
    assert fetched.started_at == now
    assert fetched.finished_at == now
    assert fetched.node_timings == {"node_a": {"duration_ms": 1200, "cached": False}}
    assert fetched.cancel_reason is None


async def test_cancel_reason_persists(db_session):
    """cancel_reason 落字符串。"""
    task = ExecutionTask(
        workflow_name="laneB-cancel",
        status="cancelled",
        cancel_reason="user requested at node sampler",
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)
    assert task.cancel_reason == "user requested at node sampler"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_execution_task_schema.py -v`
Expected: FAIL —— `TypeError: 'priority' is an invalid keyword argument for ExecutionTask`（当前 ORM 还没这些列）。

- [ ] **Step 4: 给 `ExecutionTask` 加 8 个 V1.5 列**

`backend/src/models/execution_task.py` 当前完整内容（30 行）末尾，在 `updated_at` 列定义之后、类体结尾处追加 V1.5 列。改完整个文件为：
```python
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ExecutionTask(Base):
    __tablename__ = "execution_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, default=snowflake_id)
    workflow_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    workflow_name: Mapped[str] = mapped_column(String(100), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/running/completed/failed/cancelled
    nodes_total: Mapped[int] = mapped_column(Integer, default=0)
    nodes_done: Mapped[int] = mapped_column(Integer, default=0)
    current_node: Mapped[str | None] = mapped_column(String(100), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # —— V1.5 新增（Lane B，spec §3.1）——
    # 全部 nullable，旧行保持 NULL；priority 有 default=10（batch 级），
    # 调度器入队时显式写 0（interactive）或 10（batch）。
    priority: Mapped[int] = mapped_column(Integer, default=10)
    gpu_group: Mapped[str | None] = mapped_column(String(32), nullable=True)
    runner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    queued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    node_timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_execution_task_schema.py -v`
Expected: 三个用例全 PASS。

- [ ] **Step 6: 跑 task 相关 suite 确认无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_api_execution_tasks.py tests/test_api_tasks.py -q`
Expected: PASS，通过数 = Step 1 基线（新列全 nullable / 有 default，旧的 record / get / delete 路径不传新列照常工作）。

- [ ] **Step 7: Commit**

```bash
cd backend && git add src/models/execution_task.py tests/test_execution_task_schema.py
git commit -m "feat(db): extend ExecutionTask ORM with V1.5 scheduler columns

priority/gpu_group/runner_id/queued_at/started_at/finished_at/
node_timings/cancel_reason — all nullable (priority defaults to 10),
old write paths unaffected. V1.5 Lane B, spec 3.1."
```

---

## Task 2: 手写 SQL migration（生产 PG 库增量改 schema）

ORM 加列只对「新建库 / 测试库」（`create_all`）生效。**生产 PG 库已存在 `execution_tasks` 表**，`create_all` 不会对已存在的表 `ALTER`。所以需要一份手写 migration SQL，对齐 `migrations/2026-04-23-keys-m10.sql` 的风格（单事务、`IF NOT EXISTS` 幂等）。

**Files:**
- Create: `backend/migrations/2026-05-14-v15-execution-tasks.sql`

- [ ] **Step 1: 确认既有 migration 风格**

Run: `cd backend && cat migrations/2026-04-23-keys-m10.sql`
Expected: 看到 `BEGIN; ALTER TABLE instance_api_keys ADD COLUMN IF NOT EXISTS ...; COMMIT;` 单事务幂等模式 —— 这是本 Lane migration 要对齐的模板。

- [ ] **Step 2: 写 migration 文件**

新建 `backend/migrations/2026-05-14-v15-execution-tasks.sql`：
```sql
-- backend/migrations/2026-05-14-v15-execution-tasks.sql
-- V1.5 Lane B · execution_tasks schema 扩展
-- spec: docs/superpowers/specs/2026-05-13-workflow-queue-and-gpu-scheduler-design.md §3.1
--
-- V1.5 把 image/TTS/LLM 推理从主进程串行执行改为 per-GPU-group runner 子进程
-- 调度。execution_tasks 需要 8 个新列记录调度元信息 + 可观测时间线：
--   * priority      — 2 级优先级（0=interactive / 10=batch），同级 FIFO
--   * gpu_group     — 落到哪个 hardware.yaml group（"llm-tp" / "image" / "tts"）
--   * runner_id     — 实际执行的 runner 实例 id
--   * queued_at     — 入队（DB commit）时刻；入队 sort key = (priority, queued_at)
--   * started_at    — dispatcher 弹出、标 running 的时刻
--   * finished_at   — completed/failed/cancelled 终态时刻
--   * node_timings  — 每节点耗时 JSON，每节点保留 cached:bool（V1.5 永远 false，V1.6 缓存用）
--   * cancel_reason — 取消原因（"user requested" / "node timeout" / "runner_crashed" ...）
--
-- 兼容性：8 列全部 nullable（priority 在 ORM 层有 default=10，DB 层 DEFAULT 10），
-- 旧行保持 NULL / 10，旧写入路径不传新列照常工作。
-- 单事务，IF NOT EXISTS 幂等（可重复跑）。
--
-- 部署：psql $DATABASE_URL -f backend/migrations/2026-05-14-v15-execution-tasks.sql

BEGIN;

ALTER TABLE execution_tasks
  ADD COLUMN IF NOT EXISTS priority      INTEGER NOT NULL DEFAULT 10,
  ADD COLUMN IF NOT EXISTS gpu_group     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS runner_id     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS queued_at     TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS started_at    TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS finished_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS node_timings  JSONB,
  ADD COLUMN IF NOT EXISTS cancel_reason VARCHAR(200);

-- 调度器 dispatcher 按 (priority, queued_at) 排序弹队，且启动恢复时扫 status；
-- 加一个复合索引覆盖「按 group 找排队中 task」的热路径。
CREATE INDEX IF NOT EXISTS idx_execution_tasks_sched
  ON execution_tasks (status, gpu_group, priority, queued_at);

COMMIT;
```

> 说明：ORM（Task 1）用 `JSON`，PG migration 用 `JSONB` —— SQLAlchemy 的 `JSON` 类型在 PG 后端映射为 `jsonb` 是常见做法，且 `result` 列既有数据本就是这么存的；与既有表保持一致。`priority` 在 SQL 里 `NOT NULL DEFAULT 10`，与 ORM 的 `default=10` 语义一致（ORM 侧 `Mapped[int]` 非 Optional）。

- [ ] **Step 3: SQL 语法自检（不连真库）**

Run: `cd backend && python -c "import re,sys; s=open('migrations/2026-05-14-v15-execution-tasks.sql').read(); assert s.count('BEGIN;')==1 and s.count('COMMIT;')==1, 'must be single transaction'; assert s.count('IF NOT EXISTS')==9, f'expected 9 IF NOT EXISTS (8 cols + 1 index), got {s.count(chr(39)+chr(39)) if False else s.count(\"IF NOT EXISTS\")}'; print('migration shape OK')"`
Expected: `migration shape OK`（单事务 + 8 列 + 1 索引 全部 `IF NOT EXISTS` 幂等）。

> 真库验证（部署期手动跑，不在自动化 plan 内）：`psql $DATABASE_URL -f backend/migrations/2026-05-14-v15-execution-tasks.sql` 应无错；重复跑第二次仍无错（幂等）。验证列已加：`psql $DATABASE_URL -c "\d execution_tasks"` 应看到 8 个新列。

- [ ] **Step 4: Commit**

```bash
cd backend && git add migrations/2026-05-14-v15-execution-tasks.sql
git commit -m "feat(db): SQL migration for execution_tasks V1.5 columns

Hand-written migration matching repo convention (this repo has no
alembic; see migrations/2026-04-23-keys-m10.sql). Single-transaction,
IF NOT EXISTS idempotent. Adds 8 scheduler columns + a (status,
gpu_group, priority, queued_at) composite index for the dispatcher
hot path. V1.5 Lane B, spec 3.1."
```

---

## Task 3: `TaskSnapshot` dataclass

spec §3.5 给的 `TaskSnapshot` 是 "...其余字段省略..." 的草图，明确字段是实现时定。这里把它落实成完整 dataclass —— 字段对齐 `ExecutionTask` 的可观测子集 + spec 点名的 `db_synced`。

**Files:**
- Create: `backend/src/services/task_ring_buffer.py`（本 Task 只建 `TaskSnapshot`，Task 4 加 `TaskRingBuffer`）
- Test: `backend/tests/test_task_ring_buffer.py`（新建，本 Task 先放 TaskSnapshot 用例）

- [ ] **Step 1: 写失败测试 —— TaskSnapshot 构造 + from_task 转换**

新建 `backend/tests/test_task_ring_buffer.py`：
```python
"""Lane B: TaskSnapshot + TaskRingBuffer 单元测试（纯内存，无 DB、无 GPU）。"""
from datetime import datetime, timezone

from src.services.task_ring_buffer import TaskSnapshot


def _make_snapshot(task_id: int = 1, status: str = "queued", **kw) -> TaskSnapshot:
    base = dict(
        task_id=task_id,
        workflow_name="wf",
        status=status,
        priority=10,
        gpu_group=None,
        runner_id=None,
        nodes_total=0,
        nodes_done=0,
        current_node=None,
        queued_at=None,
        started_at=None,
        finished_at=None,
        duration_ms=None,
        error=None,
        cancel_reason=None,
        db_synced=True,
    )
    base.update(kw)
    return TaskSnapshot(**base)


def test_snapshot_construct_defaults():
    snap = _make_snapshot()
    assert snap.task_id == 1
    assert snap.status == "queued"
    assert snap.db_synced is True
    assert snap.gpu_group is None


def test_snapshot_from_orm_task():
    """from_task 把 ExecutionTask ORM 行转成快照，db_synced 由调用方传入。"""

    class _FakeTask:
        # 鸭子类型，避免测试依赖真 ORM/DB
        id = 42
        workflow_name = "laneB-wf"
        status = "running"
        priority = 0
        gpu_group = "image"
        runner_id = "runner-i"
        nodes_total = 3
        nodes_done = 1
        current_node = "sampler"
        queued_at = datetime(2026, 5, 14, tzinfo=timezone.utc)
        started_at = datetime(2026, 5, 14, tzinfo=timezone.utc)
        finished_at = None
        duration_ms = None
        error = None
        cancel_reason = None

    snap = TaskSnapshot.from_task(_FakeTask(), db_synced=False)
    assert snap.task_id == 42
    assert snap.status == "running"
    assert snap.priority == 0
    assert snap.gpu_group == "image"
    assert snap.runner_id == "runner-i"
    assert snap.nodes_done == 1
    assert snap.current_node == "sampler"
    assert snap.db_synced is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_task_ring_buffer.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.services.task_ring_buffer'`。

- [ ] **Step 3: 创建 `task_ring_buffer.py` 加 `TaskSnapshot`**

新建 `backend/src/services/task_ring_buffer.py`：
```python
"""TaskRingBuffer —— 主进程内最近 200 条 task 快照的热缓存。

DB（execution_tasks 表）是真相源（survives restart）；ring buffer 是热缓存
（survives request burst，O(1) 读，给 TaskPanel / Dashboard 用）。

每条快照带 db_synced 标志：DB 写成功 → True；DB 不可达降级期写失败 → False。
该标志驱动 spec §4.6 的 reconcile（DB 恢复后批量补写 db_synced=False 的条目）。
**本 Lane（B）只提供数据结构与 db_synced 翻转 API；reconcile loop 由后续 Lane 实现。**
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

RING_CAPACITY = 200


@dataclass
class TaskSnapshot:
    """execution_tasks 一行的可观测子集 + db_synced 标志。

    字段对齐 ExecutionTask ORM（spec §3.1）的「前端 / 调度需要看」的子集；
    不含 result / node_timings 这类大载荷（TaskPanel 按需单独查 DB）。
    """

    task_id: int
    workflow_name: str
    status: str  # queued/running/completed/failed/cancelled
    priority: int
    gpu_group: str | None
    runner_id: str | None
    nodes_total: int
    nodes_done: int
    current_node: str | None
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    error: str | None
    cancel_reason: str | None
    db_synced: bool = True

    @classmethod
    def from_task(cls, task: Any, *, db_synced: bool) -> "TaskSnapshot":
        """从 ExecutionTask ORM 行（或鸭子类型等价物）构造快照。

        db_synced 必须显式传入 —— 它反映的是「这次 DB 写有没有成功」，
        是调用方（scheduler / executor）才知道的事，不是 task 行本身的属性。
        """
        return cls(
            task_id=task.id,
            workflow_name=task.workflow_name,
            status=task.status,
            priority=task.priority,
            gpu_group=task.gpu_group,
            runner_id=task.runner_id,
            nodes_total=task.nodes_total,
            nodes_done=task.nodes_done,
            current_node=task.current_node,
            queued_at=task.queued_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            duration_ms=task.duration_ms,
            error=task.error,
            cancel_reason=task.cancel_reason,
            db_synced=db_synced,
        )
```

> `field` / `replace` 在 Task 4 用到，先一并 import（Task 4 不用再改 import 行）。`from __future__ import annotations` 让 `str | None` 在 3.12 之前也安全，且与本仓库其它模块一致。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_task_ring_buffer.py -v`
Expected: `test_snapshot_construct_defaults` 和 `test_snapshot_from_orm_task` 两个 PASS。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/task_ring_buffer.py tests/test_task_ring_buffer.py
git commit -m "feat(scheduler): add TaskSnapshot dataclass for ring buffer

Observable subset of execution_tasks + db_synced flag. from_task()
converts an ORM row; db_synced is passed in by the caller (it reflects
whether the DB write succeeded, not a property of the row). V1.5
Lane B, spec 3.5."
```

---

## Task 4: `TaskRingBuffer` 类

spec §3.5 给的骨架：`_items: collections.deque[TaskSnapshot]`（maxlen=200）+ `_by_id: dict[int, TaskSnapshot]`。需要的操作：push（新 task）、update（同 task_id 状态变化，原地更新而非追加重复条目）、get（by id）、list_recent（最近 N 条）、mark_synced（翻转 db_synced，给 reconcile 用）、unsynced（列出 db_synced=False 的，给 reconcile 用）。

**Files:**
- Modify: `backend/src/services/task_ring_buffer.py`（加 `TaskRingBuffer` 类）
- Test: `backend/tests/test_task_ring_buffer.py`（追加 TaskRingBuffer 用例）

- [ ] **Step 1: 写失败测试 —— ring buffer 全部行为**

在 `backend/tests/test_task_ring_buffer.py` 末尾追加：
```python
from src.services.task_ring_buffer import RING_CAPACITY, TaskRingBuffer


def test_push_and_get():
    rb = TaskRingBuffer()
    snap = _make_snapshot(task_id=100)
    rb.push(snap)
    assert rb.get(100) is snap
    assert rb.get(999) is None
    assert len(rb) == 1


def test_update_in_place_no_duplicate():
    """同 task_id 第二次 push 应替换，不产生重复条目。"""
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, status="queued"))
    rb.push(_make_snapshot(task_id=1, status="running", nodes_done=2))
    assert len(rb) == 1
    assert rb.get(1).status == "running"
    assert rb.get(1).nodes_done == 2
    # list_recent 里也只有一条
    assert [s.task_id for s in rb.list_recent()] == [1]


def test_maxlen_evicts_oldest():
    """超过 RING_CAPACITY 后，最旧的被 evict，_by_id 同步清理。"""
    rb = TaskRingBuffer()
    for i in range(RING_CAPACITY + 5):
        rb.push(_make_snapshot(task_id=i))
    assert len(rb) == RING_CAPACITY
    # task_id 0..4 应被 evict
    for evicted in range(5):
        assert rb.get(evicted) is None
    # task_id 5..204 应还在
    assert rb.get(5) is not None
    assert rb.get(RING_CAPACITY + 4) is not None


def test_list_recent_order_and_limit():
    """list_recent 返回最近优先（新 → 旧），limit 截断。"""
    rb = TaskRingBuffer()
    for i in range(10):
        rb.push(_make_snapshot(task_id=i))
    recent = rb.list_recent(limit=3)
    assert [s.task_id for s in recent] == [9, 8, 7]
    # 不传 limit 返回全部（最近优先）
    assert [s.task_id for s in rb.list_recent()] == list(range(9, -1, -1))


def test_mark_synced_flips_flag():
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, db_synced=False))
    assert rb.get(1).db_synced is False
    ok = rb.mark_synced(1)
    assert ok is True
    assert rb.get(1).db_synced is True
    # 不存在的 id 返回 False
    assert rb.mark_synced(999) is False


def test_unsynced_lists_only_false():
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, db_synced=True))
    rb.push(_make_snapshot(task_id=2, db_synced=False))
    rb.push(_make_snapshot(task_id=3, db_synced=False))
    unsynced_ids = sorted(s.task_id for s in rb.unsynced())
    assert unsynced_ids == [2, 3]


def test_update_in_place_can_change_db_synced():
    """降级期 push(db_synced=False)，DB 恢复后 push(db_synced=True) 覆盖。"""
    rb = TaskRingBuffer()
    rb.push(_make_snapshot(task_id=1, status="running", db_synced=False))
    assert rb.get(1).db_synced is False
    rb.push(_make_snapshot(task_id=1, status="completed", db_synced=True))
    assert rb.get(1).db_synced is True
    assert rb.get(1).status == "completed"
    assert rb.unsynced() == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_task_ring_buffer.py -v`
Expected: 新追加的 7 个用例 FAIL（`ImportError: cannot import name 'TaskRingBuffer'`），Task 3 的 2 个 TaskSnapshot 用例仍 PASS。

- [ ] **Step 3: 在 `task_ring_buffer.py` 加 `TaskRingBuffer` 类**

在 `backend/src/services/task_ring_buffer.py` 末尾（`TaskSnapshot` 类之后）追加：
```python
class TaskRingBuffer:
    """最近 RING_CAPACITY 条 task 快照，O(1) push / get / mark_synced。

    deque(maxlen) 负责容量淘汰；_by_id 是 task_id → snapshot 的副本索引，
    deque 满后 evict 最旧条目时需同步从 _by_id 清掉。

    同 task_id 重复 push = 原地更新（替换 deque 中那条 + 刷新 _by_id），
    不追加重复条目 —— task 生命周期里状态会变多次（queued → running →
    completed），ring buffer 只关心「每个 task 的最新快照」。

    线程安全：本类不加锁。主进程 asyncio 单线程使用，调用方在 event loop
    内串行 push/get 即可；如需跨线程访问由调用方自行加锁（本 Lane 不涉及）。
    """

    def __init__(self) -> None:
        self._items: collections.deque[TaskSnapshot] = collections.deque(
            maxlen=RING_CAPACITY
        )
        self._by_id: dict[int, TaskSnapshot] = {}

    def push(self, snapshot: TaskSnapshot) -> None:
        """加入 / 原地更新一条快照。"""
        existing = self._by_id.get(snapshot.task_id)
        if existing is not None:
            # 原地替换 deque 中那一条（保持其位置，避免假装它是「最近」）
            idx = self._index_of(existing)
            if idx is not None:
                self._items[idx] = snapshot
            else:  # 理论不可达：_by_id 有但 deque 没有 → 当作新条目
                self._append(snapshot)
            self._by_id[snapshot.task_id] = snapshot
            return
        self._append(snapshot)

    def _append(self, snapshot: TaskSnapshot) -> None:
        """追加新条目；deque 满时 popleft 的旧条目要从 _by_id 同步清掉。"""
        if len(self._items) == RING_CAPACITY:
            evicted = self._items[0]  # 即将被 maxlen 挤掉的那条
            # 仅当 _by_id 里那条确实是被 evict 的对象时才删（防同 id 已被更新过）
            if self._by_id.get(evicted.task_id) is evicted:
                del self._by_id[evicted.task_id]
        self._items.append(snapshot)
        self._by_id[snapshot.task_id] = snapshot

    def _index_of(self, snapshot: TaskSnapshot) -> int | None:
        for i, item in enumerate(self._items):
            if item is snapshot:
                return i
        return None

    def get(self, task_id: int) -> TaskSnapshot | None:
        return self._by_id.get(task_id)

    def list_recent(self, limit: int | None = None) -> list[TaskSnapshot]:
        """最近优先（新 → 旧）。limit=None 返回全部。"""
        items = list(reversed(self._items))
        return items[:limit] if limit is not None else items

    def mark_synced(self, task_id: int) -> bool:
        """把某 task 的 db_synced 翻成 True（reconcile 补写成功后调用）。

        返回 task 是否存在于 buffer。
        """
        snap = self._by_id.get(task_id)
        if snap is None:
            return False
        synced = replace(snap, db_synced=True)
        idx = self._index_of(snap)
        if idx is not None:
            self._items[idx] = synced
        self._by_id[task_id] = synced
        return True

    def unsynced(self) -> list[TaskSnapshot]:
        """所有 db_synced=False 的快照（DB 恢复后给 reconcile 遍历补写）。"""
        return [s for s in self._items if not s.db_synced]

    def __len__(self) -> int:
        return len(self._items)
```

> 设计说明：`TaskSnapshot` 是 frozen-ish 用法但未标 `frozen=True`（`from_task` 之外也允许直接构造），`mark_synced` 用 `dataclasses.replace` 产新对象而非原地改字段 —— 保持「一次 push 一个不可变快照」的心智模型，避免别名 bug。`_index_of` 是 O(n) 线性扫，n≤200，push 频率远低于推理耗时，可接受；若后续 profiling 发现热点再换 `deque` + 位置索引（本 Lane 不预优化）。

- [ ] **Step 4: 跑测试确认全部通过**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_task_ring_buffer.py -v`
Expected: 全部 9 个用例 PASS（Task 3 的 2 个 + 本 Task 的 7 个）。

- [ ] **Step 5: Commit**

```bash
cd backend && git add src/services/task_ring_buffer.py tests/test_task_ring_buffer.py
git commit -m "feat(scheduler): add TaskRingBuffer with db_synced reconcile hooks

deque(maxlen=200) + by-id index, O(1) push/get. push() updates
in place by task_id (no duplicate entries across a task lifecycle).
mark_synced/unsynced expose the db_synced flag for the spec 4.6
DB-recovery reconcile (loop itself lands in a later Lane). V1.5
Lane B, spec 3.5."
```

---

## Task 5: Lane B 整合验证 + lint 预检

**Files:** 无（验证）

- [ ] **Step 1: Lane B 全部新测试 green**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_execution_task_schema.py tests/test_task_ring_buffer.py -v`
Expected: 3 + 9 = 12 个用例全 PASS。

- [ ] **Step 2: 后端全 suite 无回归**

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/ -q`
Expected: PASS。新增 12 个用例，无 collection error、无 import error。重点确认 `test_api_execution_tasks.py` / `test_api_tasks.py` 原有用例仍全 PASS（ORM 加 nullable 列对旧路径透明）。

- [ ] **Step 3: lint 预检（push 前本地跑）**

Run: `cd backend && ruff check src/models/execution_task.py src/services/task_ring_buffer.py tests/test_execution_task_schema.py tests/test_task_ring_buffer.py`
Expected: 无 lint 错误。

- [ ] **Step 4: 确认 migration 文件未被 ruff 之外的 hook 漏检**

Run: `cd backend && python -c "open('migrations/2026-05-14-v15-execution-tasks.sql').read(); print('migration file present + readable')"`
Expected: `migration file present + readable`。

- [ ] **Step 5: 开 PR**

```bash
cd /media/heygo/Program/projects-code/_playground/nous-center
git push -u origin <lane-B-branch>
gh pr create --title "feat: V1.5 Lane B — execution_tasks schema + TaskRingBuffer" --body "$(cat <<'EOF'
## Summary
- `ExecutionTask` ORM 加 8 个 V1.5 调度列（priority/gpu_group/runner_id/queued_at/started_at/finished_at/node_timings/cancel_reason），全部 nullable
- 手写 SQL migration `migrations/2026-05-14-v15-execution-tasks.sql`（本仓库无 alembic，对齐既有 `migrations/*.sql` 风格，单事务 IF NOT EXISTS 幂等）
- 新增 `TaskRingBuffer` + `TaskSnapshot`：200 条最近 task 快照热缓存，带 `db_synced` 标志为 §4.6 reconcile 留钩子
- 纯数据层 Lane：不接 scheduler（Lane G/I 的事）

## Test plan
- [ ] `test_execution_task_schema.py` green（新列可写可读 + 默认值 + nullable 回归）
- [ ] `test_task_ring_buffer.py` green（maxlen evict / by-id / list_recent / db_synced 翻转 / update-in-place）
- [ ] 后端全 suite green，`test_api_execution_tasks.py` 无回归
- [ ] 部署期手动：`psql $DATABASE_URL -f backend/migrations/2026-05-14-v15-execution-tasks.sql` 跑两遍均无错（幂等）
EOF
)"
```
（分支名按项目惯例，如 `feat/v15-laneB-schema-ringbuffer`。）

---

## Self-Review

**Spec 覆盖检查：** Lane B 在 spec「实施分 Lane」表里的职责是「`execution_tasks` schema migration + TaskRingBuffer（含 `db_synced` 标志）。依赖：无」。

- `execution_tasks` schema migration → Task 1（ORM 8 列）+ Task 2（SQL migration 文件）
  - 8 列与 spec §3.1 逐一核对：`priority` / `gpu_group` / `runner_id` / `queued_at` / `started_at` / `finished_at` / `node_timings` / `cancel_reason` —— 全部覆盖，类型与 spec 代码块一致（`priority` Integer default=10、`gpu_group`/`runner_id` String(32)、三个时间戳 nullable DateTime、`node_timings` JSON、`cancel_reason` String(200)）
- TaskRingBuffer + `db_synced` → Task 3（`TaskSnapshot` 含 `db_synced`）+ Task 4（`TaskRingBuffer`，`mark_synced`/`unsynced` 暴露 db_synced 给 §4.6 reconcile）
- 「依赖：无」→ 本 plan 不 import 任何 Lane 0/A 产物，纯数据层
- 「不接 scheduler」→ plan 顶部 + Task 4 注释均明确：本 Lane 只交付数据结构 + db_synced 翻转 API，reconcile loop 与 scheduler 接线是后续 Lane

**与 task 简报的偏差（已在 plan 顶部 + Task 2 显式标注）：** 简报要求 alembic migration，但**本仓库根本没有 alembic** —— 无 `alembic/`、无 `alembic.ini`。实际 schema 演进 = `Base.metadata.create_all`（建表）+ 手写 `migrations/*.sql`（增量改）。Task 2 按既有 4 份 SQL migration 的风格写，不引入 alembic（对单 admin infra 的 8 列 ADD COLUMN 引入整套迁移框架是 over-engineering，且与既有运维方式割裂）。这与 Lane 0 plan 同样在顶部 flag spec 偏差的做法一致。

**spec 模糊处的判断：**
- spec §3.5 的 `TaskSnapshot` 写的是 "...其余字段省略..." —— 字段集是实现时定。判断：取 `ExecutionTask` 的「前端 / 调度可观测子集」，**不含** `result` / `node_timings` 大载荷（TaskPanel 按需单独查 DB），含 spec 点名的 `db_synced`。已在 `TaskSnapshot` docstring 说明。
- spec §3.5 的 `TaskRingBuffer` 骨架没给方法签名 —— `push` / `get` / `list_recent` / `mark_synced` / `unsynced` 的语义是按 §4.6 reconcile 的需求 + TaskPanel「最近完成列表」的需求推出来的。`push` 取「同 task_id 原地更新」语义（task 生命周期内状态变多次，ring buffer 只关心最新快照），已在类 docstring 说明。
- spec §3.1 `priority` 在 ORM 代码块里是 `Mapped[int]`（非 Optional）`default=10`，但 Lane 表又说「所有新字段 nullable」。判断：`priority` 跟随 spec §3.1 的精确代码 —— `NOT NULL DEFAULT 10`（ORM `Mapped[int]` + SQL `NOT NULL DEFAULT 10`），其余 7 列 nullable。「所有新字段 nullable」理解为「不破坏旧写入路径」，`priority` 有 server default 同样满足这一点。已在 migration SQL 注释说明。

**Placeholder 扫描：** 无 TBD / TODO / 「类似 Task N」。所有测试代码、ORM 代码、SQL migration、`TaskRingBuffer` 实现均完整给出。每个 Task 都是「写失败测试 → 跑确认失败 → 最小实现 → 跑确认通过 → commit」闭环，命令带预期输出。

**类型一致性：**
- `ExecutionTask` 新列类型 ↔ SQL migration 列类型逐一对应：`Mapped[int] default=10` ↔ `INTEGER NOT NULL DEFAULT 10`；`Mapped[str|None] String(32)` ↔ `VARCHAR(32)`；`Mapped[datetime|None] DateTime(timezone=True)` ↔ `TIMESTAMPTZ`；`Mapped[dict|None] JSON` ↔ `JSONB`（SQLAlchemy `JSON` 在 PG 后端即 `jsonb`，与既有 `result` 列一致）；`cancel_reason String(200)` ↔ `VARCHAR(200)`。
- `TaskSnapshot.from_task` 读的字段名与 `ExecutionTask` ORM 属性名一一对应（`task.id` → `task_id` 是唯一重命名，其余同名）。
- `TaskRingBuffer.push` / `get` / `list_recent` / `mark_synced` / `unsynced` 的返回类型标注与测试断言一致（`get` → `TaskSnapshot | None`，`list_recent` → `list[TaskSnapshot]`，`mark_synced` → `bool`）。

**已知风险：**
- **SQL migration 真库未自动验证** —— `test_*` 跑的是 SQLite + `create_all`，不碰 `migrations/*.sql`。这是本仓库既有 migration（keys-m10 / passkey-totp）的固有情况，不是本 Lane 引入的。缓解：Task 2 Step 3 做了 SQL 形状自检（单事务 + 9 个 `IF NOT EXISTS`），Step 4 + PR checklist 注明部署期手动 `psql -f` 跑两遍验证幂等。
- **ORM `JSON` vs migration `JSONB`** —— 二者在 PG 上等价（既有 `result` 列就是这个组合），但 SQLite 测试库里 `JSON` 退化为 TEXT。`test_v15_columns_round_trip` 的 `node_timings` dict 往返断言能在 SQLite 上覆盖序列化正确性，PG 上的 `jsonb` 行为靠既有 `result` 列的生产实践背书。
- **`_index_of` 是 O(n) 线性扫** —— n≤200，push 频率 << 推理耗时，不构成热点。已在 Task 4 实现注释说明「后续 profiling 发现再优化，本 Lane 不预优化」。
- **`TaskRingBuffer` 不加锁** —— 设计为主进程 asyncio 单线程使用。已在类 docstring 明确「跨线程访问由调用方加锁」。后续 Lane 把它接进 scheduler 时若涉及线程桥（spec §3.3 的 pipe-reader 读线程），接线方需注意 —— 但那是 Lane C/G 的 scope，不是 Lane B。
