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

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.exceptions import NodeCancelled, QueueFullError

logger = logging.getLogger(__name__)

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
                *list(self.inflight_tasks.values()), return_exceptions=True
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
