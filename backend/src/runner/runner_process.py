"""image/TTS runner 子进程入口 + 内部双 asyncio task.

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

        # progress_callback —— 每 step 发一个 NodeProgress.
        # 本 Lane fake adapter 在 event loop 里直接调 callback,
        # 用 create_task 排发送即可（Lane G 的真 adapter callback 在 to_thread
        # 工作线程里，那时需改 loop.call_soon_threadsafe）。
        # 收集 progress 发送任务 —— 末尾在发 NodeResult 前 await 它们，
        # 保证顺序「最后一个 progress 先到、result 后到」。
        progress_tasks: list[asyncio.Task] = []

        def _on_progress(done: int, total: int, _node=node) -> None:
            t = asyncio.get_running_loop().create_task(ch.send_message(P.NodeProgress(
                task_id=_node.task_id, node_id=_node.node_id,
                progress=done / total if total else 1.0,
                detail=f"step {done}/{total}",
            )))
            progress_tasks.append(t)

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
            # 先排空 progress 发送，保证 cancelled NodeResult 在最后
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="cancelled",
                outputs=None, error="aborted",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue
        except Exception as e:  # noqa: BLE001
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        # 排空 progress 发送 —— 保证「所有 NodeProgress 先到、NodeResult 后到」
        if progress_tasks:
            await asyncio.gather(*progress_tasks, return_exceptions=True)
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
    """multiprocessing.Process 的 target —— image/TTS runner 子进程入口.

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
