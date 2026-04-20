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
        """Execute a single node."""
        node_type = node["type"]
        data = dict(node.get("data", {}))
        executor = _NODE_EXECUTORS.get(node_type)
        if executor is None:
            # Check plugin executors from node packages
            from nodes import get_all_executors
            plugin_executors = get_all_executors()
            executor = plugin_executors.get(node_type)
        if executor is None:
            raise ExecutionError(f"未知节点类型: {node_type}")
        # Inject node_id and progress callback so executors can push streaming events
        data["_node_id"] = node["id"]
        global _on_progress_ref
        _on_progress_ref = self._on_progress
        return await executor(data, inputs)


# --- Per-node executor functions ---


async def _exec_text_input(data: dict, inputs: dict) -> dict:
    return {"text": data.get("text", "")}


async def _exec_multimodal_input(data: dict, inputs: dict) -> dict:
    """Multi-modal input — outputs text and optional images."""
    # Support both single image (legacy) and multiple images
    images = data.get("images") or []
    if not images:
        single = data.get("image", "")
        if single and single.startswith("data:"):
            images = [single]
    return {
        "text": data.get("text", ""),
        "image": images[0] if images else "",  # backward compat: first image
        "images": images,
        "audio": data.get("audio_data", ""),  # base64 data URL
    }


async def _exec_ref_audio(data: dict, inputs: dict) -> dict:
    return {
        "audio_path": data.get("path", ""),
        "audio": data.get("audio_data", ""),  # base64 data URL for LLM audio input
        "ref_text": data.get("ref_text", ""),
    }


async def _exec_tts_engine(data: dict, inputs: dict) -> dict:
    """Call TTS engine via ModelManager."""
    import asyncio
    import base64

    text = inputs.get("text", "")
    if not text:
        raise ExecutionError("TTS 节点缺少文本输入")

    engine_name = data.get("engine", "cosyvoice2")

    if _model_manager is None:
        raise ExecutionError("ModelManager 未初始化")

    adapter = _model_manager.get_adapter(engine_name)
    if adapter is None or not adapter.is_loaded:
        raise ExecutionError(
            f"引擎 {engine_name} 未加载，请先通过管理 API 加载"
        )

    kwargs = {
        "text": text,
        "voice": data.get("voice", "default"),
        "speed": data.get("speed", 1.0),
        "sample_rate": data.get("sample_rate", 24000),
    }

    result = await asyncio.to_thread(adapter.synthesize, **kwargs)
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
    """Call LLM via OpenAI-compatible API, with optional streaming."""
    prompt = inputs.get("prompt") or inputs.get("text", "")
    if not prompt:
        raise ExecutionError("LLM 节点缺少 prompt 输入")

    model_key = data.get("model_key", "")
    base_url = data.get("base_url", "")
    adapter = None

    # If model_key specified, use ModelManager to resolve base_url
    if model_key and _model_manager is not None:
        adapter = _model_manager.get_adapter(model_key)
        if adapter is None or not adapter.is_loaded:
            # Try to load on demand
            await _model_manager.load_model(model_key)
            adapter = _model_manager.get_adapter(model_key)
        if adapter is not None and hasattr(adapter, "base_url"):
            base_url = adapter.base_url

    if not base_url:
        from src.config import get_settings
        base_url = get_settings().VLLM_BASE_URL

    _validate_llm_url(base_url)

    # Build messages for streaming path
    messages = []
    system_msg = data.get("system")
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    # Support multimodal: images + audio
    images = inputs.get("images") or []
    if not images:
        single = inputs.get("image") or ""
        if single and single.startswith("data:"):
            images = [single]

    audio = inputs.get("audio") or ""

    has_media = bool(images) or (audio and audio.startswith("data:"))

    if has_media:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if img and img.startswith("data:"):
                content.append({"type": "image_url", "image_url": {"url": img}})
        if audio and audio.startswith("data:"):
            content.append({"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    # Thinking mode (Qwen3.5, etc.)
    enable_thinking = str(data.get("enable_thinking", "false")).lower() == "true"

    # Clamp max_tokens to model's actual max_model_len
    max_tokens = int(data.get("max_tokens", 2048))
    model_max = 4096  # safe default
    # Try getting max_model_len from adapter (no HTTP needed)
    if adapter is not None:
        model_max = getattr(adapter, "max_model_len", model_max) or model_max
    else:
        # Fallback: query vLLM/SGLang API
        try:
            async with httpx.AsyncClient(timeout=3, proxy=None) as _c:
                _resp = await _c.get(f"{base_url.rstrip('/')}/v1/models")
                if _resp.status_code == 200:
                    models = _resp.json().get("data", [])
                    if models:
                        model_max = models[0].get("max_model_len", model_max)
        except Exception:
            pass
    safe_max = max(model_max - 512, model_max // 2)
    if max_tokens > safe_max:
        max_tokens = safe_max

    import time as _time

    # Streaming is the UX default (widget defaults to 'true'). Only skip it
    # when explicitly disabled OR no progress channel exists.
    _stream_raw = data.get("stream")
    _stream_on = _stream_raw is None or str(_stream_raw).lower() not in ("false", "0", "no", "off")
    if _stream_on and _on_progress_ref is not None:
        node_id = data.get("_node_id", "")
        on_progress = _on_progress_ref

        async def _push_token(token: str) -> None:
            await on_progress({
                "type": "node_stream",
                "node_id": node_id,
                "token": token,
            })

        params: dict[str, Any] = {
            "model": data.get("model", ""),
            "messages": messages,
            "temperature": data.get("temperature", 0.7),
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }
        if enable_thinking:
            params["chat_template_kwargs"] = {"enable_thinking": True}
        t0 = _time.monotonic()
        result = await _stream_llm(base_url, params, on_token=_push_token)
        duration_ms = int((_time.monotonic() - t0) * 1000)
        result = _strip_thinking(result)
        return {"text": result, "usage": _last_stream_usage, "duration_ms": duration_ms}

    # Non-streaming path — use raw httpx to support vision format
    extra_body: dict[str, Any] = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    body: dict[str, Any] = {
        "model": data.get("model", ""),
        "messages": messages,
        "temperature": data.get("temperature", 0.7),
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)

    headers: dict[str, str] = {}
    api_key = data.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = _time.monotonic()
    async with httpx.AsyncClient(timeout=300, proxy=None) as _client:
        resp = await _client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body, headers=headers)
        if resp.status_code != 200:
            try:
                err = resp.json()
                detail = err.get("error", {}).get("message", resp.text[:300])
            except Exception:
                detail = resp.text[:300]
            raise ExecutionError(f"LLM API error ({resp.status_code}): {detail}")
        resp_data = resp.json()
        result = resp_data["choices"][0]["message"]["content"]
    duration_ms = int((_time.monotonic() - t0) * 1000)
    result = _strip_thinking(result)
    usage = resp_data.get("usage")
    return {"text": result, "usage": usage, "duration_ms": duration_ms}


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

    _validate_llm_url(base_url)

    # Build allowed tool names set for whitelist validation
    allowed_tools = {t["function"]["name"] for t in tools} if tools else set()

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

                # Whitelist validation: reject unknown tools
                if tool_name not in allowed_tools:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: unknown tool '{tool_name}'",
                    })
                    continue

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
        try:
            matched = bool(re.search(condition, text))
        except re.error:
            raise ExecutionError(f"无效的正则表达式: {condition}")
    elif match_type == "not_empty":
        matched = bool(text.strip())
    else:
        matched = False

    return {"true": text if matched else "", "false": text if not matched else ""}


async def _exec_text_output(data: dict, inputs: dict) -> dict:
    """Display text — passes through input text."""
    return {"text": inputs.get("text", "")}


_NODE_EXECUTORS = {
    "text_input": _exec_text_input,
    "text_output": _exec_text_output,
    "multimodal_input": _exec_multimodal_input,
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
