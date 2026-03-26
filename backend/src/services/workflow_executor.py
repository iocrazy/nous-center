"""Backend DAG workflow executor."""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any

from src.services import agent_manager
from src.services.llm_service import call_llm

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    pass


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
                await self._on_progress({
                    "type": "node_complete",
                    "node_id": node_id,
                    "step": i + 1,
                    "total": total,
                    "progress": round(((i + 1) / total) * 100),
                })

        return {"outputs": self._outputs}

    async def _execute_node(self, node: dict, inputs: dict) -> dict[str, Any]:
        """Execute a single node."""
        node_type = node["type"]
        data = node.get("data", {})
        executor = _NODE_EXECUTORS.get(node_type)
        if executor is None:
            # Check plugin executors from node packages
            from nodes import get_all_executors
            plugin_executors = get_all_executors()
            executor = plugin_executors.get(node_type)
        if executor is None:
            raise ExecutionError(f"未知节点类型: {node_type}")
        return await executor(data, inputs)


# --- Per-node executor functions ---


async def _exec_text_input(data: dict, inputs: dict) -> dict:
    return {"text": data.get("text", "")}


async def _exec_ref_audio(data: dict, inputs: dict) -> dict:
    return {
        "audio_path": data.get("path", ""),
        "ref_text": data.get("ref_text", ""),
    }


async def _exec_tts_engine(data: dict, inputs: dict) -> dict:
    """Call TTS engine via the engine registry."""
    import asyncio
    import base64

    from src.workers.tts_engines import registry

    text = inputs.get("text", "")
    if not text:
        raise ExecutionError("TTS 节点缺少文本输入")

    engine_name = data.get("engine", "cosyvoice2")
    engine = registry._ENGINE_INSTANCES.get(engine_name)
    if not engine or not engine.is_loaded:
        raise ExecutionError(
            f"引擎 {engine_name} 未加载，请先通过管理 API 加载"
        )

    kwargs = {
        "text": text,
        "voice": data.get("voice", "default"),
        "speed": data.get("speed", 1.0),
        "sample_rate": data.get("sample_rate", 24000),
    }

    result = await asyncio.to_thread(engine.synthesize, **kwargs)
    audio_b64 = base64.b64encode(result.audio_bytes).decode()
    return {
        "audio": audio_b64,
        "sample_rate": result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "format": result.format,
    }


async def _exec_output(data: dict, inputs: dict) -> dict:
    return inputs


async def _exec_passthrough(data: dict, inputs: dict) -> dict:
    """Stub for unimplemented audio processing nodes."""
    return inputs


async def _exec_llm(data: dict, inputs: dict) -> dict:
    """Call LLM via OpenAI-compatible API."""
    from src.services import model_scheduler

    prompt = inputs.get("prompt") or inputs.get("text", "")
    if not prompt:
        raise ExecutionError("LLM 节点缺少 prompt 输入")

    model_key = data.get("model_key", "")
    base_url = data.get("base_url", "")

    # If model_key specified, use scheduler to get base_url
    if model_key:
        url = model_scheduler.get_llm_base_url(model_key)
        if url:
            base_url = url
        else:
            # Try to load on demand
            await model_scheduler.load_model(model_key)
            url = model_scheduler.get_llm_base_url(model_key)
            if url:
                base_url = url
            else:
                raise ExecutionError(f"模型 {model_key} 未加载")

    if not base_url:
        base_url = "http://localhost:8100"

    result = await call_llm(
        prompt=prompt,
        base_url=base_url,
        model=data.get("model", ""),
        system=data.get("system"),
        api_key=data.get("api_key"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2048),
    )
    return {"text": result}


async def _exec_prompt_template(data: dict, inputs: dict) -> dict:
    """Replace {variable} placeholders in a template with input values."""
    template = data.get("template", "")
    result = template
    for key, value in inputs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return {"text": result}


async def _exec_agent(data: dict, inputs: dict) -> dict:
    """Execute agent with multi-turn tool call loop."""
    from src.services.llm_service import call_llm_with_tools
    from src.services.skill_tools import skills_to_tools, execute_tool
    from src.config import get_settings

    agent_name = data.get("agent_name", "")
    input_text = inputs.get("text", "")
    if not agent_name:
        raise ExecutionError("Agent 节点未选择 Agent")
    if not input_text:
        raise ExecutionError("Agent 节点缺少输入")

    agent = agent_manager.get_agent(agent_name)
    if not agent:
        raise ExecutionError(f"Agent '{agent_name}' 不存在")

    # Assemble system prompt from MD files
    prompts = agent.get("prompts", {})
    system_parts: list[str] = []
    for fname in ["IDENTITY.md", "SOUL.md", "AGENT.md"]:
        content = prompts.get(fname, "").strip()
        if content:
            system_parts.append(content)
    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # Get tools from agent's skills
    agent_skills = agent.get("skills", [])
    tools = skills_to_tools(agent_skills)

    # LLM config
    model_config = agent.get("model", {})
    settings = get_settings()
    base_url = model_config.get("base_url") or settings.VLLM_BASE_URL
    model = model_config.get("model") or model_config.get("engine_key") or ""
    api_key = model_config.get("api_key") or model_config.get("fallback_api")

    # Multi-turn loop (max 10 iterations to prevent infinite loops)
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})

    max_iterations = 10
    response: dict = {}
    for _i in range(max_iterations):
        response = await call_llm_with_tools(
            messages=messages,
            base_url=base_url,
            model=model,
            api_key=api_key,
            tools=tools if tools else None,
        )

        # Check if response has tool calls
        if response.get("tool_calls"):
            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response["tool_calls"],
            })

            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                try:
                    result = await execute_tool(tool_name, tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: {e}",
                    })
        else:
            # No tool calls — return final response
            return {"text": response.get("content", "")}

    # Max iterations reached
    return {"text": response.get("content", "Agent 达到最大迭代次数")}


async def _exec_python_code(data: dict, inputs: dict) -> dict:
    """Execute Python code from the node."""
    from src.services.skill_tools import _execute_python

    code = data.get("code", "")
    if not code and inputs.get("text"):
        code = inputs["text"]
    result = await _execute_python(code)
    return {"text": result}


async def _exec_if_else(data: dict, inputs: dict) -> dict:
    """Conditional branching based on match_type."""
    condition = data.get("condition", "")
    match_type = data.get("match_type", "contains")
    text = inputs.get("text", "")

    if match_type == "contains":
        matched = condition in text
    elif match_type == "equals":
        matched = text == condition
    elif match_type == "regex":
        matched = bool(re.search(condition, text))
    elif match_type == "not_empty":
        matched = bool(text.strip())
    else:
        matched = False

    return {"true": text if matched else "", "false": text if not matched else ""}


_NODE_EXECUTORS = {
    "text_input": _exec_text_input,
    "ref_audio": _exec_ref_audio,
    "tts_engine": _exec_tts_engine,
    "output": _exec_output,
    "resample": _exec_passthrough,
    "mixer": _exec_passthrough,
    "concat": _exec_passthrough,
    "bgm_mix": _exec_passthrough,
    "llm": _exec_llm,
    "prompt_template": _exec_prompt_template,
    "agent": _exec_agent,
    "if_else": _exec_if_else,
    "python_exec": _exec_python_code,
}
