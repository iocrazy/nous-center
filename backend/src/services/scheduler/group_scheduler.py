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
