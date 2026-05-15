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
