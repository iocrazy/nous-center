"""RunnerClient —— 主进程侧、PipeChannel 之上的节点级 RPC（spec §3.5）.

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
import time
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

        # PR-5a §5: ComponentEvent 回调 —— 主进程订阅组件加载状态迁移
        self.on_component_event: Callable[[P.ComponentEvent], None] | None = None

        self._demux_task: asyncio.Task | None = None

        # Lane K follow-up: inflight dispatch 状态跟踪 —— 给 /api/v1/monitor/runners
        # 的 current_task 字段供数据，让前端 TaskPanel 真显示「正在跑啥」+ 进度条。
        # task_id -> {task_id, workflow_name, node_id, node_type, started_at, progress, detail}
        self._dispatches: dict[int, dict[str, Any]] = {}

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def current_dispatch(self) -> dict[str, Any] | None:
        """当前 runner 正在执行的任务信息 —— 给 /api/v1/monitor/runners 的 current_task
        字段供数据。多个 inflight 时优先返回有 progress 的(= runner 在它身上发过
        NodeProgress = 正在跑它);都没 progress 则返回最早入队的(probably-next-to-run)。
        无 inflight 返回 None。
        """
        if not self._dispatches:
            return None
        progressing = [d for d in self._dispatches.values() if d.get("progress", 0) > 0]
        if progressing:
            return max(progressing, key=lambda d: d.get("progress", 0))
        return min(self._dispatches.values(), key=lambda d: d.get("started_at", 0))

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
        """runner 断连 —— 所有等待中的 future 置异常 + dispatch 跟踪清空。"""
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
        self._dispatches.clear()

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
                # Lane K follow-up: 把最新 progress 写进 dispatch 状态,/runners 暴露
                d = self._dispatches.get(msg.task_id)
                if d is not None:
                    d["progress"] = msg.progress
                    d["detail"] = msg.detail
                cb = self._progress_cbs.get(msg.task_id)
                if cb is not None:
                    cb(msg)
            elif isinstance(msg, P.NodeResult):
                fut = self._node_futures.pop(msg.task_id, None)
                self._progress_cbs.pop(msg.task_id, None)
                # Lane K follow-up: dispatch 完成,从跟踪表移除
                self._dispatches.pop(msg.task_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
            elif isinstance(msg, P.ComponentEvent):
                cb = self.on_component_event
                if cb is not None:
                    cb(msg)
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
        workflow_name: str = "",
    ) -> P.NodeResult:
        """发 RunNode，await 对应的 NodeResult。

        workflow_name: 给 /api/v1/monitor/runners current_task 显示用的人读字段
        (前端 TaskPanel 泳道挂在 runner 上的「正在跑」名字)。无则空串。
        """
        if not self._connected:
            raise ConnectionError("runner disconnected")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._node_futures[spec.task_id] = fut
        if on_progress is not None:
            self._progress_cbs[spec.task_id] = on_progress
        # Lane K follow-up: 登记 inflight dispatch,供 current_dispatch property 读
        self._dispatches[spec.task_id] = {
            "task_id": spec.task_id,
            "workflow_name": workflow_name,
            "node_id": spec.node_id,
            "node_type": spec.node_type,
            "started_at": time.monotonic(),
            "progress": 0.0,
            "detail": None,
        }
        try:
            await self._ch.send_message(spec)
            return await fut
        finally:
            # 兜底:_fail_all_inflight / NodeResult 都会移除,但异常路径(send_message 抛、
            # 取消)也保证不残留。
            self._dispatches.pop(spec.task_id, None)

    async def load_model(self, model_key: str, *, config: dict | None = None) -> bool:
        """发 LoadModel，await ModelEvent。返回是否加载成功。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._model_futures[model_key] = fut
        await self._ch.send_message(P.LoadModel(model_key=model_key, config=config or {}))
        return await fut

    async def preload_components(
        self,
        task_id: int,
        components: dict,
        pipeline_class: str = "Flux2KleinPipeline",
    ) -> None:
        """发 PreloadComponents —— fire-and-forget；状态走 ComponentEvent → on_component_event。"""
        if not self._connected:
            raise ConnectionError("runner disconnected")
        await self._ch.send_message(
            P.PreloadComponents(
                task_id=task_id,
                components=components,
                pipeline_class=pipeline_class,
            )
        )

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
