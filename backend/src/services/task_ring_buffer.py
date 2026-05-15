"""TaskRingBuffer —— 主进程内最近 200 条 task 快照的热缓存。

DB（execution_tasks 表）是真相源（survives restart）；ring buffer 是热缓存
（survives request burst，O(1) 读，给 TaskPanel / Dashboard 用）。

每条快照带 db_synced 标志：DB 写成功 → True；DB 不可达降级期写失败 → False。
该标志驱动 spec §4.6 的 reconcile（DB 恢复后批量补写 db_synced=False 的条目）。
**本 Lane（B）只提供数据结构与 db_synced 翻转 API；reconcile loop 由后续 Lane 实现。**
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, replace
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
