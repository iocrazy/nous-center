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
