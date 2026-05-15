"""image/TTS runner 子进程入口 + 内部双 asyncio task.

spec §4.4 / D9：runner 子进程内跑两个 task：
  * pipe-reader —— 持续读 pipe：RunNode 入内部 asyncio.Queue；Abort 置对应
    task 的 threading.Event；LoadModel/UnloadModel/Ping 直接处理。永不阻塞在
    adapter 上 —— 这样 Abort 才能立即置位。
  * node-executor —— 从队列取 RunNode、ModelManager.get_or_load adapter、
    调 adapter.infer (可选传 progress_callback + cancel_flag，按签名探测)、
    发 NodeProgress / NodeResult。

cancel 信号用 threading.Event：真 adapter 的扩散循环在 to_thread 里跑，跨线程
信号必须用 threading 原语（spec §4.4 关键性质 D14）。本文件 Lane D 阶段：fake
adapter 支持 progress_callback + cancel_flag；真 image adapter 的
`infer(req)` 还不接这俩 kwarg（Lane G/D14 才接）。node-executor 用 signature
探测决定是否传。

Lane D：每个 runner 子进程持有独立 ModelManager（spec §4.5）。LoadModel /
UnloadModel / RunNode 全走 ModelManager。
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from typing import Any

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel


class _RunnerState:
    """runner 子进程内的可变状态。"""

    def __init__(
        self,
        runner_id: str,
        group_id: str,
        gpus: list[int],
        model_manager,  # src.services.model_manager.ModelManager
    ):
        self.runner_id = runner_id
        self.group_id = group_id
        self.gpus = gpus
        # Lane D：真 ModelManager（per-runner 独立实例，spec §4.5）。
        # 替换 Lane C 的极简 dict[model_key -> adapter]。
        self.mm = model_manager
        # 待执行的 RunNode 队列（pipe-reader 投，node-executor 取）
        self.run_queue: asyncio.Queue[P.RunNode] = asyncio.Queue()
        # task_id -> cancel flag（pipe-reader 收 Abort 时 set）
        self.cancel_flags: dict[int, threading.Event] = {}
        # task_id 收到 Abort 但 RunNode 尚未到达 —— RunNode 到达时把这个标记翻成
        # 立即置位的 cancel_flag。覆盖「先 Abort 后 RunNode」 / 「RunNode 还在
        # pipe-reader 队列里就收到 Abort」两种节点边界 cancel 时序（spec §4.4）。
        self.pending_aborts: set[int] = set()
        self.shutdown = asyncio.Event()


def _merge_config_into_spec(state: _RunnerState, model_key: str, config: dict) -> None:
    """把 LoadModel.config 合并进该 model 的 ModelSpec.params。

    ModelSpec frozen —— 用 model_copy(update=...) 不可变更新。真实部署 config
    一般空；这条路径主要服务测试通过 LoadModel 注入 fake 故障开关
    （oom_on_load_count / fail_load / infer_seconds）。
    """
    if not config:
        return
    spec = state.mm._registry.get(model_key)
    if spec is None:
        return
    merged = {**spec.params, **config}
    state.mm._registry._specs[model_key] = spec.model_copy(
        update={"params": merged}
    )


async def _handle_load_model(state: _RunnerState, ch: PipeChannel, msg: P.LoadModel) -> None:
    """LoadModel —— 走 ModelManager.get_or_load（含 OOM evict 重试），发 ModelEvent。"""
    from src.errors import ModelLoadError, ModelNotFoundError

    _merge_config_into_spec(state, msg.model_key, msg.config)
    try:
        await state.mm.get_or_load(msg.model_key)
    except (ModelLoadError, ModelNotFoundError) as e:
        await ch.send_message(P.ModelEvent(
            event="load_failed", model_key=msg.model_key,
            error=f"{type(e).__name__}: {e}",
        ))
        return
    except Exception as e:  # noqa: BLE001 —— 兜底，runner 不崩
        await ch.send_message(P.ModelEvent(
            event="load_failed", model_key=msg.model_key,
            error=f"{type(e).__name__}: {e}",
        ))
        return
    await ch.send_message(P.ModelEvent(event="loaded", model_key=msg.model_key, error=None))


async def _handle_unload_model(state: _RunnerState, ch: PipeChannel, msg: P.UnloadModel) -> None:
    await state.mm.unload_model(msg.model_key, force=True)
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
            flag = threading.Event()
            # 先 Abort 后 RunNode（或 Abort 紧跟 RunNode 但还没 race 完）：
            # pending_aborts 里有同 task_id → 立即置位 flag，让 node-executor
            # 在 dispatch 前 boundary check 直接判 cancelled（spec §4.4）。
            if msg.task_id in state.pending_aborts:
                flag.set()
                state.pending_aborts.discard(msg.task_id)
            state.cancel_flags[msg.task_id] = flag
            state.run_queue.put_nowait(msg)
        elif isinstance(msg, P.Abort):
            flag = state.cancel_flags.get(msg.task_id)
            if flag is not None:
                flag.set()  # node-executor 的 adapter 下一 step 边界看到
            else:
                # Abort 先到 / RunNode 还没到 —— 记下，RunNode 到了再合并
                state.pending_aborts.add(msg.task_id)
        elif isinstance(msg, P.LoadModel):
            await _handle_load_model(state, ch, msg)
        elif isinstance(msg, P.UnloadModel):
            await _handle_unload_model(state, ch, msg)
        elif isinstance(msg, P.Ping):
            await ch.send_message(P.Pong(
                runner_id=state.runner_id,
                loaded_models=list(state.mm.loaded_model_ids),
            ))
        # 其余消息类型（runner→主进程方向的）不应收到，忽略


def _build_request(node: P.RunNode):
    """按 node_type 构造 typed InferenceRequest。

    spec §3.3：RunNode.node_type 仅 "image" / "tts"。image 走 ImageRequest
    （within-node cancel，progress callback per step）；tts 走 AudioRequest
    （spec §4.4：boundary-cancel only，infer(req) 不接 progress/cancel kwargs）。
    未知 node_type 抛 ValueError —— node-executor 转成 NodeResult status=failed。
    """
    from src.services.inference.base import AudioRequest, ImageRequest

    if node.node_type == "image":
        return ImageRequest(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps", 1) or 1),
        )
    if node.node_type == "tts":
        return AudioRequest(
            request_id=f"task-{node.task_id}",
            text=str(node.inputs.get("text", "")),
            voice=str(node.inputs.get("voice", "default")),
            speed=float(node.inputs.get("speed", 1.0) or 1.0),
            sample_rate=int(node.inputs.get("sample_rate", 24000) or 24000),
        )
    raise ValueError(f"unsupported node_type {node.node_type!r} (expected image / tts)")


async def _node_executor(state: _RunnerState, ch: PipeChannel) -> None:
    """从队列取 RunNode，跑 adapter，发 progress / result。"""
    from src.errors import ModelLoadError, ModelNotFoundError

    while not state.shutdown.is_set():
        try:
            node = await asyncio.wait_for(state.run_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue  # 周期性回头看 shutdown

        cancel_flag = state.cancel_flags.get(node.task_id) or threading.Event()
        started = time.monotonic()

        # ModelManager.get_or_load —— per-model lock + OOM evict + load failure 检查
        try:
            adapter = await state.mm.get_or_load(node.model_key) if node.model_key else None
        except (ModelLoadError, ModelNotFoundError) as e:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        if adapter is None:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=f"node {node.node_id!r} has no model_key",
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue

        # progress_callback —— 每 step 发一个 NodeProgress.
        # 本 Lane fake adapter 在 event loop 里直接调 callback,
        # 用 create_task 排发送即可（Lane G 的真 adapter callback 在 to_thread
        # 工作线程里，那时需改 loop.call_soon_threadsafe）。
        progress_tasks: list[asyncio.Task] = []

        def _on_progress(done: int, total: int, _node=node) -> None:
            t = asyncio.get_running_loop().create_task(ch.send_message(P.NodeProgress(
                task_id=_node.task_id, node_id=_node.node_id,
                progress=done / total if total else 1.0,
                detail=f"step {done}/{total}",
            )))
            progress_tasks.append(t)

        try:
            req = _build_request(node)
            if node.node_type == "tts":
                # spec §4.4：TTS = boundary-cancel only。TTSEngine.infer(req)
                # 签名只收 req —— 不传 progress_callback / cancel_flag。节点边界
                # 的 cancel 由 pipe-reader 在 dispatch 前置位的 cancel_flag +
                # 下面的 boundary check 覆盖（含 Abort-before-RunNode）。
                if cancel_flag.is_set():
                    raise asyncio.CancelledError()
                result = await adapter.infer(req)
            else:
                # image：用 signature 探测决定是否传 progress_callback / cancel_flag。
                # FakeAdapter 接受这俩 kwarg；真 image adapter 由 Lane G/D14 接
                # callback_on_step_end + CancelFlag。
                infer_params = inspect.signature(adapter.infer).parameters
                infer_kwargs: dict = {}
                if "progress_callback" in infer_params:
                    infer_kwargs["progress_callback"] = _on_progress
                if "cancel_flag" in infer_params:
                    infer_kwargs["cancel_flag"] = cancel_flag
                result = await adapter.infer(req, **infer_kwargs)
        except ValueError as e:
            # 未知 node_type —— _build_request 抛 ValueError。明确判 failed，
            # 不崩 runner。注意：必须放在泛 except Exception 之前，否则被吞掉、
            # 错误信息里就没有 "node_type" 字样。
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=str(e),
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue
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
    models_yaml_path: str | None = None,
    fake_adapter: bool = False,
) -> None:
    """multiprocessing.Process 的 target —— image/TTS runner 子进程入口。

    起独立 event loop（spec §4.5：runner 有自己的 Event Loop B）+ 构造 per-runner
    独立 ModelManager（spec §4.5）。fake_adapter=True → 所有模型走 FakeAdapter。
    """
    from src.runner.runner_modelmanager import build_runner_model_manager

    runner_id = f"runner-{group_id}-{uuid.uuid4().hex[:6]}"
    mm = build_runner_model_manager(
        group_id, gpus, models_yaml_path=models_yaml_path, fake_adapter=fake_adapter,
    )
    state = _RunnerState(runner_id, group_id, gpus, mm)
    ch = PipeChannel(conn)
    try:
        asyncio.run(_runner_loop(state, ch))
    finally:
        ch.close()
