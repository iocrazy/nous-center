"""Backend DAG workflow executor."""

from __future__ import annotations

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
    "image_generate": "image",
    "tts_engine": "tts",
}


def set_model_manager(mgr: ModelManager) -> None:
    global _model_manager
    _model_manager = mgr


class ExecutionError(Exception):
    pass


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

    def _topological_sort(self) -> list[str]:
        if not self.nodes:
            raise ExecutionError("工作流为空")

        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            in_degree.setdefault(node["id"], 0)

        for edge in self.edges:
            adj[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1

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
        """Collect inputs for a node from upstream outputs via edges."""
        inputs: dict[str, Any] = {}
        for edge in self.edges:
            if edge["target"] == node_id:
                source_output = self._outputs.get(edge["source"], {})
                source_handle = edge.get("sourceHandle", "")
                target_handle = edge.get("targetHandle", "")
                if source_handle in source_output:
                    inputs[target_handle] = source_output[source_handle]
                for key, value in source_output.items():
                    if key not in inputs:
                        inputs[key] = value
        return inputs

    async def execute(self) -> dict[str, Any]:
        """Execute the workflow and return all node outputs."""
        order = self._topological_sort()
        total = len(order)

        for i, node_id in enumerate(order):
            node = self._node_map[node_id]
            inputs = self._get_inputs(node_id)

            if self._on_progress:
                await self._on_progress({
                    "type": "node_start",
                    "node_id": node_id,
                    "node_type": node["type"],
                    "step": i + 1,
                    "total": total,
                    "progress": round((i / total) * 100),
                })

            try:
                output = await self._run_node_routed(node, inputs)
                self._outputs[node_id] = output
            except Exception as e:
                if self._on_progress:
                    await self._on_progress({
                        "type": "node_error",
                        "node_id": node_id,
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
                    if "duration_ms" in output:
                        complete_event["duration_ms"] = output["duration_ms"]
                await self._on_progress(complete_event)

        return {"outputs": self._outputs}

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

        spec = P.RunNode(
            task_id=task_id,
            node_id=node["id"],
            node_type=node["type"],
            model_key=model_key,
            inputs=inputs,
        )
        result = await client.run_node(spec, workflow_name=self._workflow_name)
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

        当前 dispatch 节点白名单很短(image_generate / tts_engine)、role 与
        group_id 一一对应；映射写在这里:image_generate→"image" / tts_engine→"tts"。
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
            async def _on_token(token: str) -> None:
                if self._on_progress:
                    await self._on_progress({
                        "type": "node_stream",
                        "node_id": node["id"],
                        "content": token,
                    })

            result = await instance.stream(data, inputs, _on_token)
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
