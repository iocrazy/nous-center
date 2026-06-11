"""Backend DAG workflow executor."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Any

from src.services import agent_manager  # noqa: F401 — test-patching seam (AgentNode reads via we.agent_manager)
from src.services.llm_service import call_llm  # noqa: F401 — re-exported for test patching
from src.services.model_manager import ModelManager

# Trigger @register side effects for all builtin nodes
from src.services.nodes import audio, image, llm, logic, text_io  # noqa: F401

EVENT_TYPES: tuple[str, ...] = (
    # Existing events
    "node_start",
    "node_stream",
    "node_complete",
    "node_error",
    "complete",
    # Wave 1 new events (coze-style)
    "node_end_streaming",        # 流式最后一个 chunk 发出后触发（vs node_complete 是逻辑完成点）
    "workflow_interrupt",        # QA 节点等需要 human-in-the-loop 时触发（本波只占位，不实现节点）
    "workflow_resume",           # 从 interrupt 恢复时触发
    "function_call",             # LLM 发起 tool call 时触发（预留 tool-use 事件）
    "tool_response",             # tool 返回结果
    "tool_streaming_response",   # tool 流式返回
)

logger = logging.getLogger(__name__)

_model_manager: ModelManager | None = None
_on_progress_ref = None

# Lane K: dispatch 节点 → runner group_id 路由表。group_id 约定与
# hardware.yaml 的 role 名一致(image/tts/llm)。新增 GPU 节点必须在此登记。
# 与 src.services.node_routing.DISPATCH_NODE_TYPES 配对维护。
_NODE_TYPE_TO_GROUP_ID: dict[str, str] = {
    "flux2_vae_decode": "image",  # 细粒度图 dispatch 终端(spec 2026-05-21 rev 2)
    "tts_engine": "tts",
    "seedvr2_upscale": "image",  # 图→图超分,跑 image GPU 组(SeedVR2 PR-3b)
}

# fine workflow type → runner 请求 role(= RunNode.node_type,runner_process._build_request
# 据此构 typed request)。**多数与 group_id 同名**,但 SeedVR2 例外:它跑在 image GPU 组
# (group_id=image,复用 image runner)却需构 UpscaleRequest(role="upscale",非 ImageRequest)
# —— 所以 role 与 group_id 在此解耦。未登记的 type 回退 group_id(与历史行为一致)。
_NODE_TYPE_TO_RUNNER_ROLE: dict[str, str] = {
    "flux2_vae_decode": "image",
    "tts_engine": "tts",
    "seedvr2_upscale": "upscale",
}


def set_model_manager(mgr: ModelManager) -> None:
    global _model_manager
    _model_manager = mgr


class ExecutionError(Exception):
    pass


class _LlmProgressEmitter:
    """PR-4 重写:LLM stream token-rate L3 进度发射器 — 委托共享 ProgressTracker。

    抽取共享逻辑(latency 滑窗 / ETA / 250ms throttle)到 ProgressTracker(PR-4)。
    LLM 的 on_progress 是 async(dict-shape WS payload),不能在 ProgressTracker 的 sync
    callback 里 await — 所以用 `callback=None` 模式:pt.step()/finish() 返 payload,
    本类再 await on_progress(包装成 node_progress dict)。throttle 跳过时 pt 返 None。
    """

    def __init__(
        self, *,
        node_id: str, max_tokens: int, on_progress: Any,
        stage: str = "llm_gen",
    ) -> None:
        from src.services.inference.progress_tracker import ProgressTracker  # noqa: PLC0415

        self._node_id = node_id
        self._max_tokens = max(1, int(max_tokens))
        self._on_progress = on_progress
        self._stage = stage
        self._token_count = 0
        # callback=None 模式:pt 不直接 emit,只计算 latency/ETA/throttle 返 payload。
        self._pt = ProgressTracker(
            None, stage=stage, throttle_ms=250, latency_window=16,
        )

    async def _emit_payload(self, payload: dict | None) -> None:
        """把 pt 算出来的 payload 包装成 node_progress dict 再 await on_progress。
        payload=None 时(被 throttle 跳过)什么都不做。"""
        if payload is None or self._on_progress is None:
            return
        await self._on_progress({
            "type": "node_progress",
            "node_id": self._node_id,
            "step": payload["done"],
            "total_steps": payload["total"],
            **{k: v for k, v in payload.items()
               if k not in ("done", "total") and v is not None},
        })

    async def on_token(self, _token: str) -> None:
        if self._on_progress is None:
            return
        self._token_count += 1
        payload = self._pt.step(self._token_count, self._max_tokens)
        await self._emit_payload(payload)

    async def emit_final(self, *, true_completion: int | None) -> None:
        """stream 完成后发末帧:step / total_steps 回填到 true_completion(usage 给的;
        没给就用本地 token 计数),progress=1.0 / eta=0。前端 callout 从「47/2048」
        变成「47/47 ✓」(显示真实生成长度,不再用估值上限)。"""
        if self._on_progress is None:
            return
        total = true_completion if true_completion is not None else self._token_count
        payload = self._pt.finish(total, detail=f"{self._stage} done ({total} tokens)")
        await self._emit_payload(payload)


class WorkflowExecutor:
    """Execute a workflow DAG (topological sort + per-node execution)."""

    def __init__(self, workflow: dict, on_progress=None, runner_client=None,
                 runner_clients: dict | None = None,
                 task_id: int | None = None, workflow_name: str = ""):
        self.nodes: list[dict] = workflow.get("nodes", [])
        self.edges: list[dict] = workflow.get("edges", [])
        self._node_map: dict[str, dict] = {n["id"]: n for n in self.nodes}
        self._outputs: dict[str, dict[str, Any]] = {}
        self._on_progress = on_progress  # async callback(data: dict)
        # Lane C RunnerClient（spec §3.3）；inline-only workflow 可传 None。
        # 出现 dispatch 节点但 runner_client=None 时，_dispatch_node 显式报错，
        # 绝不静默在主进程 inline 跑 GPU 节点（那正是 V1.5 要消灭的 GPU race）。
        #
        # Lane K: runner_clients (dict group_id → client) 是新的多 group 入口 ——
        # 节点按 type → role → group_id 选 client。runner_client (单数) 为兼容旧
        # 调用方保留:有它就当 catch-all (任何 dispatch 节点都用它)。两者都给:
        # runner_clients 优先,runner_client 作 fallback。
        self._runner_client = runner_client
        self._runner_clients: dict = runner_clients or {}
        # Lane K follow-up: 给 RunnerClient.run_node 传 task_id + workflow_name,
        # supervisor.health_snapshot.current_task 才能正确显示「在跑哪个 task」。
        self._task_id = task_id
        self._workflow_name = workflow_name
        # Bug 1(节点高亮错位):flux2_vae_decode 是 dispatch 终端,整条 Load→Encode→
        # KSampler→VAE 链的 GPU 活(加载/denoise/vae)都在它内部跑 —— 若按 node_start
        # 一律高亮 dispatch 节点,蓝边就永远糊在 VAE Decode 上(用户:"在 load 模型却聚焦
        # 到 vae")。改成按 runner 回传的 stage 把高亮"走链"到对应画布节点。这两个字段
        # 在一次 dispatch 期间记录 stage→节点映射 + 当前点亮的节点。
        self._cur_stage_walk: dict | None = None
        self._active_stage_node: str | None = None

    def _topological_sort(self) -> list[str]:
        if not self.nodes:
            raise ExecutionError("工作流为空")

        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            in_degree.setdefault(node["id"], 0)

        for edge in self.edges:
            s, t = edge["source"], edge["target"]
            # round5:校验两端都在 node_map。早先对任意 target 都 in_degree += 1,即使该 id
            # 不在 nodes(前端删节点没清对应 edge = 常见)→ 幽灵节点进 order 使 len(order)
            # > len(nodes) → 末尾 `len(order) != len(nodes)` **误报「循环依赖」**(实为悬空边),
            # 或 execute() 里 _node_map[ghost] KeyError。显式报可定位的错。
            if s not in self._node_map or t not in self._node_map:
                raise ExecutionError(f"边引用了不存在的节点: {s} → {t}(删节点后未清理连线?)")
            adj[s].append(t)
            in_degree[t] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ExecutionError("工作流存在循环依赖")

        return order

    def _get_inputs(self, node_id: str) -> dict[str, Any]:
        """Collect inputs for a node from upstream outputs via edges.

        round2 #8:有显式 sourceHandle 的边只精确路由该 handle → targetHandle,**不再**额外把
        上游全部输出键 spread 进来 —— 旧逻辑每条边无条件 spread,导致多入边同名键互相覆盖
        (先到先得)+ 精确连线被无关键污染。无 handle 的老连线仍保留 spread fallback(向后兼容)。
        """
        inputs: dict[str, Any] = {}
        for edge in self.edges:
            if edge["target"] != node_id:
                continue
            source_output = self._outputs.get(edge["source"], {})
            source_handle = edge.get("sourceHandle", "")
            target_handle = edge.get("targetHandle", "")
            if source_handle and source_handle in source_output:
                # 显式 handle:精确路由,不 spread(避免污染 + 同名覆盖)
                inputs[target_handle] = source_output[source_handle]
            else:
                # 无 handle 的边:fallback 把上游全部输出灌进来(兼容老 handle-less 连线)
                for key, value in source_output.items():
                    if key not in inputs:
                        inputs[key] = value
                # 图像 bundle 多路输入(image_url 但 handle 名非输出键,如 sourceHandle="image"
                # → 输出键是 image_url):额外把 image_url 也按 target_handle 落一份,让**多个**
                # 图输入(ColorMatch 的 image_target/image_ref)各自拿到区分 URL —— 否则上面
                # spread 的 image_url 被首条边占位、后续边 `key not in inputs` 跳过 → 第二路图丢失。
                # 纯加法:单图消费者仍读 image_url(spread 不变),零回归。
                if target_handle and "image_url" in source_output:
                    inputs.setdefault(target_handle, source_output["image_url"])
        return inputs

    async def execute(self) -> dict[str, Any]:
        """Execute the workflow and return all node outputs."""
        order = self._topological_sort()
        total = len(order)

        for i, node_id in enumerate(order):
            node = self._node_map[node_id]
            inputs = self._get_inputs(node_id)

            # 节点旁路(对齐 ComfyUI bypass):跳过执行,把上游 inputs 原样作为 outputs
            # 透传 —— 同名 handle(LoRA 的 model/clip 进出同名)直通下游;类型变化节点
            # (Encode CLIP→CONDITIONING)下游 _get_inputs 找不到对应 handle → 走兜底,
            # 下游缺该输入(= ComfyUI 旁路语义,不在此报错)。flag 存 node.data.bypassed,
            # 旁路节点执行前即 skip → 该 key 永不进 merge_inputs / ImageRequest。
            if node.get("data", {}).get("bypassed"):
                self._outputs[node_id] = dict(inputs)
                if self._on_progress:
                    await self._on_progress({
                        "type": "node_complete",
                        "node_id": node_id,
                        "step": i + 1,
                        "total": total,
                        "progress": round(((i + 1) / total) * 100),
                        "bypassed": True,
                    })
                continue

            # Bug 1:image dispatch 节点高亮"走链" —— node_start 落到加载节点(而非一律
            # 落 VAE dispatch 终端);后续按 stage 在 _forward_progress 里迁移。
            stage_walk = self._compute_image_stage_walk(node)
            self._cur_stage_walk = stage_walk
            self._active_stage_node = stage_walk["initial"] if stage_walk else None

            if self._on_progress:
                start_node_id = stage_walk["initial"] if stage_walk else node_id
                await self._on_progress({
                    "type": "node_start",
                    "node_id": start_node_id,
                    "node_type": self._node_map.get(start_node_id, {}).get("type", node["type"]),
                    "step": i + 1,
                    "total": total,
                    "progress": round((i / total) * 100),
                })

            # 逐组件时长(VAE decode 真计时,spec 2026-06-04):image dispatch 把 dit_denoise /
            # vae_decode 各自耗时塞 InferenceResult.metadata.stage_latency_ms,经 runner outputs["meta"]
            # 流到这里。据此让 KSampler 显**纯 denoise**、VAE Decode(dispatch 终端节点)显**真
            # decode 时长**(修「VAE Decode 恒 0s」—— 旧版 dec 无 duration_ms 兜底算 0)。
            stage_lat: dict[str, Any] = {}
            try:
                output = await self._run_node_routed(node, inputs)
                self._outputs[node_id] = output
                if isinstance(output, dict) and isinstance(output.get("meta"), dict):
                    _sl = output["meta"].get("stage_latency_ms")
                    if isinstance(_sl, dict):
                        stage_lat = _sl
                # stage-walk:dispatch 收尾时 active 可能停在非 VAE 节点(如末 stage 没发
                # vae_decode)→ 补完成它;VAE dispatch 节点由下方通用 node_complete 收口。
                if stage_walk and self._active_stage_node and self._active_stage_node != node_id and self._on_progress:
                    evt: dict[str, Any] = {"type": "node_complete", "node_id": self._active_stage_node}
                    dit_ms = stage_lat.get("dit_denoise")
                    if isinstance(dit_ms, (int, float)):
                        evt["duration_ms"] = int(dit_ms)  # KSampler 显纯 denoise
                    await self._on_progress(evt)
                self._cur_stage_walk = None
                self._active_stage_node = None
            except Exception as e:
                # round5:dispatch 失败时优先把错落到真正失败的 stage 节点(如 text_encode/
                # KSampler),而非 dispatch 终端(VAE Decode)—— 与第四轮修的「高亮走链」一致,
                # 否则蓝边高亮在 Encode、红错却落 VAE。在重置 _active_stage_node 前捕获它。
                err_node_id = self._active_stage_node or node_id
                self._cur_stage_walk = None
                self._active_stage_node = None
                if self._on_progress:
                    await self._on_progress({
                        "type": "node_error",
                        "node_id": err_node_id,
                        "error": str(e),
                    })
                raise ExecutionError(
                    f"节点 {node_id} ({node['type']}) 执行失败: {e}"
                ) from e

            if self._on_progress:
                complete_event: dict = {
                    "type": "node_complete",
                    "node_id": node_id,
                    "step": i + 1,
                    "total": total,
                    "progress": round(((i + 1) / total) * 100),
                }
                if isinstance(output, dict):
                    if "usage" in output:
                        complete_event["usage"] = output["usage"]
                    # VAE decode 真计时:dispatch 终端(flux2_vae_decode)节点的时长 = decode-only
                    # (stage_lat.vae_decode);无则回退老的 output["duration_ms"](tts 等)。
                    dec_ms = stage_lat.get("vae_decode")
                    if isinstance(dec_ms, (int, float)):
                        complete_event["duration_ms"] = int(dec_ms)
                    elif "duration_ms" in output:
                        complete_event["duration_ms"] = output["duration_ms"]
                    if output.get("cached"):
                        complete_event["cached"] = True
                    # Lane S 异步:图像结果(image_url 等)随 node_complete 带回前端
                    # ImageOutputNode 据此显示预览(旧同步 unwrapOutputs 已移除——它在
                    # /execute 的 202 响应上必崩 "reading 'out'")。
                    if output.get("image_url"):
                        for _k in ("image_url", "image_urls", "media_type", "width", "height",
                                   "seed", "steps", "cfg_scale"):
                            if output.get(_k) is not None:
                                complete_event[_k] = output[_k]
                await self._on_progress(complete_event)

        return {"outputs": self._outputs}

    def _ancestors(self, node_id: str) -> set[str]:
        """node_id 的全部上游祖先节点 id(沿 edges 反向 BFS)。用于把 stage 映射到
        *这条链上* 的 encode/ksampler/load 节点,不误选别的链。"""
        rev: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            rev[e["target"]].append(e["source"])
        seen: set[str] = set()
        stack = list(rev.get(node_id, []))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(rev.get(n, []))
        return seen

    def _compute_image_stage_walk(self, node: dict) -> dict | None:
        """Bug 1:对 flux2_vae_decode dispatch 节点,算出 stage→画布节点 id 的高亮映射。

        runner 回传的 NodeProgress.stage ∈ {text_encode, dit_denoise, vae_decode};把它们
        分别映射回链上的 Encode Prompt / KSampler / VAE Decode 节点,加载阶段(无 stage 事件)
        先点亮 Load Diffusion Model 节点。找不到某节点则回退到 dispatch 节点本身(行为同旧版,
        高亮落 VAE,不崩)。非 flux2_vae_decode(tts/llm/inline)返 None —— 不改其行为。"""
        if node.get("type") != "flux2_vae_decode":
            return None
        anc = self._ancestors(node["id"])

        def _find(types: set[str]) -> str | None:
            for nid in anc:
                if self._node_map.get(nid, {}).get("type") in types:
                    return nid
            return None

        vae = node["id"]
        load_node = _find({"flux2_load_diffusion_model", "flux2_load_checkpoint"})
        enc_node = _find({"flux2_encode_prompt"})
        ks_node = _find({"flux2_ksampler"})
        return {
            "targets": {
                "text_encode": enc_node or vae,
                "dit_denoise": ks_node or vae,
                "vae_decode": vae,
            },
            "initial": load_node or enc_node or vae,  # 加载阶段先点亮的节点
            "vae": vae,
        }

    async def _run_node_routed(self, node: dict, inputs: dict) -> dict[str, Any]:
        """按节点类型分流：inline 节点主进程内 await，dispatch 节点投 RunnerClient。

        spec §2.1 step 9 / §4.5「Inline 执行点改道清单」。
        """
        from src.services.node_routing import node_exec_class

        if node_exec_class(node["type"]) == "dispatch":
            return await self._dispatch_node(node, inputs)
        return await self._execute_inline_node(node, inputs)

    async def _dispatch_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """GPU 节点 → RunnerClient.run_node（spec §3.3 RunNode/NodeResult RPC）。

        runner_client 缺失时显式报错 —— 绝不静默在主进程内 inline 跑 GPU 节点
        （那正是 V1.5 要消灭的 GPU race）。

        Lane K：先按 node_type → role → group_id 在 runner_clients dict 里挑;
        没命中再 fallback 到单数 runner_client（兼容老调用方）。
        构造 protocol.RunNode 传给 client.run_node —— 注意:client.run_node 签名
        是 (spec: P.RunNode, *, on_progress=..., workflow_name=...),不是 (node, inputs)。
        """
        from src.runner import protocol as P

        client = self._pick_runner_client(node)
        if client is None:
            raise ExecutionError(
                f"节点 {node['id']} ({node['type']}) 需要 GPU runner，"
                f"但 executor 未注入 runner_client / runner_clients"
            )

        # task_id: 没有 outer ExecutionTask 时(inline-only test 路径)用 node hash
        # 做唯一 int —— 避免协议层 task_id=None 崩;UI current_task 仍能正确分。
        task_id = self._task_id if self._task_id is not None else abs(hash(node["id"])) % (2**31)
        # model_key:从 node.data 拿(LLM/TTS 有 engine/model_key 字段;image 节点 model
        # 由 adapter 自决,这里 None 也合法)。
        data = node.get("data", {})
        model_key = data.get("model_key") or data.get("engine") or data.get("model")
        # protocol §3.3:RunNode.node_type 是 role("image" / "tts"),不是
        # workflow 画布的 type("flux2_vae_decode" / "tts_engine") —— runner_process
        # ._build_request 按 role 分流构造 ImageRequest / AudioRequest。
        # _NODE_TYPE_TO_GROUP_ID 顶部维护的映射本就把 workflow type → role,这里复用。
        group_id = _NODE_TYPE_TO_GROUP_ID.get(node["type"])
        if group_id is None:
            raise ExecutionError(
                f"节点 {node['id']} 的 type {node['type']!r} 不在 dispatch 节点白名单"
            )
        # merge node.data → inputs:节点本地配置(steps/width/height/loras/cfg_scale/seed)
        # 和上游 edges 传过来的 inputs 合并。inputs 后写覆盖 data —— 上游 text_input
        # 给 prompt 时盖掉 data.prompt 是 inline 路径的语义(see image.py:49)。
        # 不带本步,runner 端 ImageRequest 拿不到 steps/width,会用 Field default
        # 25/1024,但 loras / cfg_scale / seed 全丢。
        merged_inputs = {**{k: v for k, v in data.items() if not k.startswith("_")}, **inputs}
        # spec §3.3: seed 非空 ⇒ 确定性,runner / L2 cache 据此决定可缓存。
        # round5:细粒度图终端 flux2_vae_decode 只有 vae+latent 口,**没有顶层 seed** ——
        # seed 是 KSampler widget、被 exec_ksampler 塞进嵌套 latent["seed"]。只看顶层
        # 会让带固定 seed 的出图恒非确定性 → L2 输出缓存对唯一图像路径永久失效。两处都认。
        def _has_seed(d: dict[str, Any]) -> bool:
            if d.get("seed") not in (None, ""):
                return True
            latent = d.get("latent")
            return isinstance(latent, dict) and latent.get("seed") not in (None, "")
        is_deterministic = _has_seed(merged_inputs)

        # RunNode.node_type 是 runner 请求 role(image/tts/upscale),多数 = group_id,
        # SeedVR2 例外(group_id=image 但 role=upscale)—— 见 _NODE_TYPE_TO_RUNNER_ROLE。
        runner_role = _NODE_TYPE_TO_RUNNER_ROLE.get(node["type"], group_id)
        spec = P.RunNode(
            task_id=task_id,
            node_id=node["id"],
            node_type=runner_role,
            model_key=model_key,
            inputs=merged_inputs,
            is_deterministic=is_deterministic,
        )
        # PR-3:转发 runner 的 NodeProgress → WS 节点级进度(KSampler/VAE Decode 的 callback_on_step_end
        # 每步发一个 NodeProgress;前端 DeclarativeNode 按 node_id 渲染进度条)。
        loop = asyncio.get_running_loop()
        on_progress_async = self._on_progress
        # _forward_progress(demux 同步回调)里 create_task 排发的进度/高亮事件全收进这里,
        # run_node 返回后统一 gather —— 否则 fire-and-forget 的 task 可能排在 execute() 已
        # await 的 node_complete/complete 之后(高亮乱序),且 workflow 收尾时 pending 触发
        # "Task destroyed but pending" 告警 + 丢帧(#3 修复)。
        progress_tasks: list[asyncio.Task] = []

        def _forward_progress(pmsg: "P.NodeProgress") -> None:
            if on_progress_async is None:
                return
            # Bug 1:按 stage 把高亮 + 进度重定向到链上对应画布节点(text_encode→Encode
            # Prompt / dit_denoise→KSampler / vae_decode→VAE Decode),而非一律糊在 dispatch
            # 终端。stage 切换时 complete 上一个、start 当前,蓝边随真实执行阶段"走链"。
            target_node_id = pmsg.node_id
            sw = self._cur_stage_walk
            stage = getattr(pmsg, "stage", None)
            if sw is not None and stage in sw["targets"]:
                target_node_id = sw["targets"][stage]
                if target_node_id != self._active_stage_node:
                    prev = self._active_stage_node
                    if prev is not None and prev != target_node_id:
                        progress_tasks.append(loop.create_task(on_progress_async(
                            {"type": "node_complete", "node_id": prev})))
                    progress_tasks.append(loop.create_task(on_progress_async({
                        "type": "node_start", "node_id": target_node_id,
                        "node_type": self._node_map.get(target_node_id, {}).get("type"),
                    })))
                    self._active_stage_node = target_node_id
            event: dict[str, Any] = {
                "type": "node_progress",
                "node_id": target_node_id,
                "progress": pmsg.progress,
                "detail": pmsg.detail,
            }
            # PR-F:latent 预览 thumbnail(data URI)透传到 WS,前端节点上叠图。
            if getattr(pmsg, "preview_url", None):
                event["preview_url"] = pmsg.preview_url
            # PR-1a / PR-1b:L3 stage 字段(image: text_encode / dit_denoise / vae_decode;
            # tts: tts_synth)+ step + ETA 透传。前端 ActiveTaskRow / callout 据此渲染
            # 「⚡ dit step 27/50 · ETA 5.5s」/ 「🔊 合成 3/6秒 · ETA 3s」(spec §State model
            # TaskProgress)。任一字段为 None 不发,保前端解析时只看有的键。
            for field_name in ("stage", "step", "total_steps", "step_latency_ms", "eta_ms"):
                v = getattr(pmsg, field_name, None)
                if v is not None:
                    event[field_name] = v
            progress_tasks.append(loop.create_task(on_progress_async(event)))
            # PR-6:同时广播到全局 /ws/tasks,带 task_id 路由。前端 GlobalTopbar 单一 WS
            # 连接收所有 task 的 L3 progress,ActiveTaskRow 按 task.id 匹配 — 多任务并发
            # 场景每行 callout 都能拿到自己的 stage/step/ETA。
            if self._task_id is not None:
                from src.api.websocket import ws_manager  # noqa: PLC0415
                progress_tasks.append(
                    loop.create_task(ws_manager.broadcast_task_progress(self._task_id, event)))

        # Bug 2(RUNNING 无运行进度):模型加载阶段在 denoise 前,无 step 可报 —— 任务面板
        # RUNNING 卡此期间一片空白(用户截图就是 Flux2 加载阶段)。dispatch 前先发一个
        # stage=model_load 的不定态任务进度,让 ActiveTaskRow 立刻显示「加载模型中…」。
        # 无 step/不造假百分比(见 #196 删假进度教训);真 step 进度一来即被 denoise 覆盖。
        if self._task_id is not None and self._cur_stage_walk is not None:
            from src.api.websocket import ws_manager  # noqa: PLC0415
            await ws_manager.broadcast_task_progress(self._task_id, {
                "type": "node_progress",
                "node_id": self._cur_stage_walk["initial"],
                "stage": "model_load",
                "progress": 0.0,
            })

        result = await client.run_node(
            spec, on_progress=_forward_progress, workflow_name=self._workflow_name)
        # 排空进度/高亮事件 —— 保证它们先于 execute() 随后 await 的 node_complete 到达
        # (高亮不乱序),且不留 pending task 到 workflow 收尾(#3 修复)。run_node 已返回
        # = 所有 NodeProgress 已投递,这些 task 都已创建。
        if progress_tasks:
            await asyncio.gather(*progress_tasks, return_exceptions=True)
        # 真 RunnerClient 返回 P.NodeResult dataclass;Lane S FakeRunnerClient
        # 直接返回 outputs dict(stub 简化)。统一在这里 unwrap —— executor 上层
        # 把结果当 dict 走(_outputs 索引、下游 _get_inputs 迭代),不 unwrap 就
        # 'NodeResult' is not iterable 炸。failed 状态显式抛,让 execute() 包装
        # 成 ExecutionError + 发 node_error 事件,跟 inline 节点失败一致。
        if isinstance(result, P.NodeResult):
            if result.status != "completed":
                raise RuntimeError(result.error or f"node {result.node_id} {result.status}")
            return result.outputs or {}
        return result

    def _pick_runner_client(self, node: dict):
        """按 node_type → role → group_id 在 runner_clients dict 里挑 RunnerClient。

        当前 dispatch 节点白名单很短(flux2_vae_decode / tts_engine)、role 与
        group_id 一一对应；映射写在这里:flux2_vae_decode→"image" / tts_engine→"tts"。
        新增 dispatch 节点要在此登记。runner_clients 命中失败 → fallback
        到单数 runner_client (向后兼容)。
        """
        node_type = node.get("type", "")
        # node_type → group_id (按 hardware.yaml 的 role 名作 id 约定)。
        group_id = _NODE_TYPE_TO_GROUP_ID.get(node_type)
        if group_id is not None:
            client = self._runner_clients.get(group_id)
            if client is not None:
                return client
        return self._runner_client

    async def _execute_inline_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """Execute a single node via registered class + protocol dispatch."""
        from src.services.nodes.base import InvokableNode, StreamableNode
        from src.services.nodes.registry import get_node_class

        node_type = node["type"]
        data = dict(node.get("data", {}))
        data["_node_id"] = node["id"]

        node_cls = get_node_class(node_type)
        if node_cls is None:
            # Plugin executors are still legacy functions — keep the old
            # _on_progress_ref shim alive for them during the transition.
            from nodes import get_all_executors
            plugin_executors = get_all_executors()
            legacy_fn = plugin_executors.get(node_type)
            if legacy_fn is None:
                raise ExecutionError(f"未知节点类型: {node_type}")
            global _on_progress_ref
            _on_progress_ref = self._on_progress
            return await legacy_fn(data, inputs)

        instance = node_cls()

        if isinstance(instance, StreamableNode) and data.get("stream") is not False:
            # PR-1c(2026-05-27 任务面板重置 L3 LLM lane):per-token node_stream + throttled
            # node_progress(stage=llm_gen,step=tokens,total_steps=max_tokens,
            # step_latency_ms=平均/token,eta_ms)。前端 ActiveTaskRow callout 显示
            # 「🤖 gen 47/2048 · 23 tok/s · ETA 1.5m」。throttle 250ms 防止高速 stream
            # 把 WS 打爆(60+ token/s 时若每 token 都发 progress 是 60+ msg/s 冗余)。
            # PR-1d(vision lane):node_type=llm 且 inputs 含图/音 → stage="vision_inference"
            # (vllm OpenAI 兼容接口下,vision encode + 生成在 server 端不可分,所以共享 LLM
            # 路径,只换 stage 标识让前端 callout 用 vision 配色 + Vision 图标渲染)。
            max_tokens = int(data.get("max_tokens") or 2048)  # round5:空串 widget → 默认,不 int("") 崩
            from src.services.nodes.llm import _has_multimodal_input  # PR-1d:同模块的多模态探测
            stage = ("vision_inference"
                     if node_type == "llm" and _has_multimodal_input(inputs)
                     else "llm_gen")
            emitter = _LlmProgressEmitter(
                node_id=node["id"], max_tokens=max_tokens,
                on_progress=self._on_progress, stage=stage)

            async def _on_token(token: str) -> None:
                if self._on_progress:
                    await self._on_progress({
                        "type": "node_stream",
                        "node_id": node["id"],
                        "content": token,
                    })
                await emitter.on_token(token)

            result = await instance.stream(data, inputs, _on_token)
            # 末帧:真值回填 —— tokens 用 usage.completion_tokens(若有)否则用本地计数。
            true_completion = None
            if isinstance(result, dict):
                u = result.get("usage") or {}
                true_completion = u.get("completion_tokens")
            await emitter.emit_final(true_completion=true_completion)
            if self._on_progress:
                await self._on_progress({
                    "type": "node_end_streaming",
                    "node_id": node["id"],
                    "usage": result.get("usage") if isinstance(result, dict) else None,
                })
            return result

        if isinstance(instance, InvokableNode):
            return await instance.invoke(data, inputs)

        raise ExecutionError(
            f"Node class for {node_type!r} implements neither InvokableNode nor StreamableNode"
        )
