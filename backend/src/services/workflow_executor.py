"""Backend DAG workflow executor."""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections import defaultdict, deque
from typing import Any

import httpx

from src.services import agent_manager
from src.services.llm_service import call_llm  # noqa: F401 — re-exported for test patching
from src.services.model_manager import ModelManager
from src.utils.constants import ALLOWED_LLM_HOSTS

# Trigger @register side effects for all builtin nodes
from src.services.nodes import audio, llm, logic, text_io  # noqa: F401

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


def set_model_manager(mgr: ModelManager) -> None:
    global _model_manager
    _model_manager = mgr


_last_stream_usage: dict | None = None


async def _stream_llm(base_url: str, params: dict, on_token=None) -> str:
    """Stream LLM response, calling on_token for each chunk. Returns full text.
    Captures final usage in module-level _last_stream_usage (include_usage)."""
    global _last_stream_usage
    import json as _json

    full_text = ""
    _last_stream_usage = None
    async with httpx.AsyncClient(timeout=300, proxy=None) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={**params, "stream": True},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                try:
                    err = _json.loads(body)
                    detail = err.get("error", {}).get("message", body.decode()[:200])
                except Exception:
                    detail = body.decode()[:200]
                raise ExecutionError(f"LLM API error ({resp.status_code}): {detail}")

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = _json.loads(payload)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token and on_token:
                            await on_token(token)
                        full_text += token
                    usage = chunk.get("usage")
                    if usage:
                        _last_stream_usage = usage
                except Exception:
                    pass
    return full_text


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output, return only the final answer."""
    result = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    return result.strip()


class ExecutionError(Exception):
    pass


def _validate_llm_url(url: str) -> str:
    """Ensure LLM base_url only points to localhost."""
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and parsed.hostname not in ALLOWED_LLM_HOSTS:
        raise ExecutionError(f"LLM base_url 只允许 localhost，收到: {parsed.hostname}")
    return url


class WorkflowExecutor:
    """Execute a workflow DAG (topological sort + per-node execution)."""

    def __init__(self, workflow: dict, on_progress=None):
        self.nodes: list[dict] = workflow.get("nodes", [])
        self.edges: list[dict] = workflow.get("edges", [])
        self._node_map: dict[str, dict] = {n["id"]: n for n in self.nodes}
        self._outputs: dict[str, dict[str, Any]] = {}
        self._on_progress = on_progress  # async callback(data: dict)

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
                output = await self._execute_node(node, inputs)
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

    async def _execute_node(self, node: dict, inputs: dict) -> dict[str, Any]:
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
