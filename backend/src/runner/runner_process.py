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
        from src.services.inference.image_l2_cache import ImageOutputCache
        self.image_l2 = ImageOutputCache()


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


def _make_component_event_sender(ch: PipeChannel):
    """返回 async on_event(component_key, state, error)，把 ComponentEvent 写入
    pipe —— 传给 ModelManager.get_or_load_image_adapter，让组件状态变迁到达
    backend（spec §6.1）。"""
    async def _on_event(component_key: str, state: str, error: str | None = None) -> None:
        await ch.send_message(P.ComponentEvent(component_key=component_key, state=state, error=error))
    return _on_event


async def _handle_preload_components(state: _RunnerState, ch: PipeChannel, msg: P.PreloadComponents) -> None:
    """PreloadComponents → get_or_load_image_adapter（发 ComponentEvent）。不抛 ——
    失败已通过 ComponentEvent(state=failed) 报告，runner 不能崩。"""
    from src.services.inference.component_spec import ComponentSpec
    try:
        components = {k: ComponentSpec(**v) for k, v in msg.components.items()}
    except Exception as e:  # noqa: BLE001 — bad descriptor
        await ch.send_message(P.ComponentEvent(component_key="?", state="failed", error=f"bad spec: {e}"))
        return
    on_event = _make_component_event_sender(ch)
    try:
        await state.mm.get_or_load_image_adapter(components, msg.pipeline_class, on_event=on_event)
    except Exception:  # noqa: BLE001 — 已通过 on_event 逐组件上报，不再二次抛
        pass


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
        elif isinstance(msg, P.PreloadComponents):
            await _handle_preload_components(state, ch, msg)
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
        from src.services.inference.component_spec import ComponentSpec

        # 细粒度图 dispatch 终端(flux2_vae_decode):inputs 带嵌套 latent + vae。
        # 摊平成 ImageRequest;整模型单卡 —— clip/vae 的 device 强制 = unet 的 device
        # (Load Diffusion Model 上选的卡;clip/vae 跟随)。spec 2026-05-21 rev 2。
        latent = node.inputs.get("latent")
        vae_d = node.inputs.get("vae")
        if (isinstance(latent, dict) and latent.get("_type") == "flux2_latent"
                and isinstance(vae_d, dict) and vae_d.get("_type") == "flux2_vae"):
            model_d = latent["model"]
            cond_d = latent["conditioning"]
            unet_spec = dict(model_d["spec"])
            device = unet_spec["device"]
            encoders = cond_d["clip"]["encoders"]
            if len(encoders) != 1:
                raise ValueError(
                    f"多编码器 CLIP({len(encoders)} 条 encoder)执行 PR-3 才支持;"
                    f"当前单编码器(flux2/qwen)")
            clip_spec = dict(encoders[0])
            clip_spec["device"] = device
            vae_spec = dict(vae_d["spec"])
            vae_spec["device"] = device
            lseed = latent.get("seed")
            return ImageRequest(
                request_id=f"task-{node.task_id}",
                prompt=str(cond_d.get("text", "")),
                negative_prompt=str(cond_d.get("negative", "")),
                width=int(latent.get("width") or 1024),
                height=int(latent.get("height") or 1024),
                steps=int(latent.get("steps") or 25),
                cfg_scale=float(latent.get("cfg_scale") or 4.0),
                seed=int(lseed) if lseed not in (None, "") else None,
                components={
                    "unet": ComponentSpec(loras=model_d.get("loras") or [], **unet_spec),
                    "clip": ComponentSpec(**clip_spec),
                    "vae": ComponentSpec(**vae_spec),
                },
                pipeline_class="Flux2KleinPipeline",
            )

        # 转发 ImageRequest 全部字段 —— executor 已把 node.data 合并进 inputs，
        # steps/width/height/cfg_scale/seed/loras 都在这里能拿到。缺省走 pydantic
        # Field default(steps=25 / 1024x1024 / cfg_scale=7.0)。
        raw_seed = node.inputs.get("seed")
        seed = int(raw_seed) if raw_seed not in (None, "") else None
        base = dict(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps") or 25),
            width=int(node.inputs.get("width") or 1024),
            height=int(node.inputs.get("height") or 1024),
            cfg_scale=float(node.inputs.get("cfg_scale") or 7.0),
            seed=seed,
        )
        # 新格式：三组件描述符齐全 → 走 components 路径（spec §5.4）。
        if all(k in node.inputs for k in ("unet", "clip", "vae")):
            return ImageRequest(
                **base,
                components={
                    "unet": ComponentSpec(**node.inputs["unet"]),
                    "clip": ComponentSpec(**node.inputs["clip"]),
                    "vae":  ComponentSpec(**node.inputs["vae"]),
                },
                pipeline_class=str(node.inputs.get("pipeline_class") or "Flux2KleinPipeline"),
            )
        # 老路径：无 components（workflow_executor 已 inline 展开过；走不到也安全）。
        loras_raw = node.inputs.get("loras") or []
        return ImageRequest(**base, loras=loras_raw if isinstance(loras_raw, list) else [])
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
    while not state.shutdown.is_set():
        try:
            node = await asyncio.wait_for(state.run_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue  # 周期性回头看 shutdown

        cancel_flag = state.cancel_flags.get(node.task_id) or threading.Event()
        started = time.monotonic()

        # 先 build typed request —— components 路径据此分流 adapter 获取方式。
        try:
            req = _build_request(node)
        except ValueError as e:
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=str(e),
                duration_ms=int((time.monotonic() - started) * 1000)))
            state.cancel_flags.pop(node.task_id, None)
            continue

        # PR-6: L2 output cache —— 确定性 image 节点二跑命中则跳过 load+infer。
        l2_key = None
        if node.node_type == "image" and getattr(node, "is_deterministic", False):
            from src.services.inference.image_l2_cache import image_l2_key, serve_image_l2
            l2_key = image_l2_key(node, req)
            entry = state.image_l2.get(l2_key)
            if entry is not None:
                ttl = int(node.inputs.get("url_ttl_seconds") or 3600)
                hit = serve_image_l2(entry, ttl)
                if hit is not None:
                    await ch.send_message(P.NodeResult(
                        task_id=node.task_id, node_id=node.node_id, status="completed",
                        outputs={
                            "meta": hit["meta"], "media_type": hit["media_type"],
                            "image_url": hit["image_url"], "image_uuid": hit["image_uuid"],
                            "image_expires": hit["image_expires"],
                            "width": hit["width"], "height": hit["height"], "cached": True,
                        },
                        error=None,
                        duration_ms=int((time.monotonic() - started) * 1000)))
                    state.cancel_flags.pop(node.task_id, None)
                    continue
                # PNG reaped → drop stale entry, fall through to recompute
                state.image_l2._d.pop(l2_key, None)

        # adapter 获取:components 路径走 get_or_load_image_adapter(组件级 L1 +
        # combo 缓存);否则老 model_key 路径(get_or_load,含 OOM evict)。
        try:
            components = getattr(req, "components", None)
            if components:
                adapter = await state.mm.get_or_load_image_adapter(
                    components, getattr(req, "pipeline_class", "Flux2KleinPipeline"),
                    on_event=_make_component_event_sender(ch))
            else:
                adapter = await state.mm.get_or_load(node.model_key) if node.model_key else None
        except Exception as e:  # noqa: BLE001
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
                outputs=None, error=f"node {node.node_id!r} has no model_key / components",
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
        # outputs payload —— 与 inline image_generate/tts 节点对齐(spec §3.3 +
        # workflow_publish exposed_outputs 白名单)。image 走 write_image 落盘签
        # URL(NAS_OUTPUTS_PATH + ADMIN_SESSION_SECRET HMAC),把 image_url 塞进
        # outputs,下游 image_output 节点才能从 inputs.image_url 取到。把 bytes
        # 通过 msgpack pipe 直接传 50MB 是反模式。
        outputs: dict[str, Any] = {"meta": result.metadata, "media_type": result.media_type}
        if node.node_type == "image" and result.media_type.startswith("image/") and result.data:
            from src.services.image_output_storage import write_image
            ext = result.media_type.split("/", 1)[1].split("+", 1)[0] or "png"
            ttl = int(node.inputs.get("url_ttl_seconds") or 3600)
            record = write_image(result.data, ext=ext, ttl_seconds=ttl)
            meta = result.metadata or {}
            outputs.update({
                "image_url": record["url"],
                "image_uuid": record["uuid"],
                "image_expires": record["expires"],
                "width": meta.get("width"),
                "height": meta.get("height"),
            })
            if l2_key is not None:
                state.image_l2.put(l2_key, {
                    "image_uuid": record["uuid"], "date": record["date"], "ext": ext,
                    "meta": result.metadata, "width": meta.get("width"), "height": meta.get("height"),
                })
        await ch.send_message(P.NodeResult(
            task_id=node.task_id, node_id=node.node_id, status="completed",
            outputs=outputs,
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
