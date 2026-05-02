"""LLM node — InvokableNode (non-stream) + StreamableNode (stream).

Calls the v2 InferenceAdapter directly via ModelManager.get_loaded_adapter:
adapter owns httpx, base_url, max_model_len clamp, and SSE parsing. The node
just builds a TextRequest, awaits infer / iterates infer_stream, and returns
the text + usage shape the executor / frontend expects.
"""

from __future__ import annotations

import logging
import time as _time
import urllib.parse
import uuid
from typing import Any

from src.services.inference.base import (
    InferenceAdapter,
    Message,
    StreamEvent,
    TextRequest,
)
from src.services.nodes.base import OnTokenFn
from src.services.nodes.registry import register
from src.utils.constants import ALLOWED_LLM_HOSTS

logger = logging.getLogger(__name__)


class _LLMExecutionError(Exception):
    """Internal error raised by LLMNode. WorkflowExecutor wraps these."""


def _validate_llm_url(url: str) -> str:
    """Defense-in-depth: external LLM base_urls (AgentNode) must be localhost.

    The v2 vLLM/SGLang adapters validate their own base_url; this helper
    remains for AgentNode which calls call_llm_with_tools against an
    operator-supplied URL outside the adapter pipeline.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and parsed.hostname not in ALLOWED_LLM_HOSTS:
        raise _LLMExecutionError(
            f"LLM base_url 只允许 localhost，收到: {parsed.hostname}"
        )
    return url


def _strip_thinking(text: str) -> str:
    import re
    result = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    return result.strip()


def _build_messages(data: dict, inputs: dict) -> list[Message]:
    """Build typed Message list from inputs.

    Priority:
    1. inputs['messages'] is a list → coerce each entry to Message.
    2. Build from data['system'] + inputs['prompt'/'text'] + multimodal
       (images/audio).
    """
    supplied = inputs.get("messages")
    if isinstance(supplied, list) and supplied:
        return [Message.model_validate(m) if isinstance(m, dict) else m for m in supplied]

    prompt = inputs.get("prompt") or inputs.get("text", "")
    if not prompt:
        raise _LLMExecutionError("LLM 节点缺少 prompt 输入")

    msgs: list[Message] = []
    system_msg = data.get("system")
    if system_msg:
        msgs.append(Message(role="system", content=system_msg))

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
            content.append(
                {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}
            )
        msgs.append(Message(role="user", content=content))
    else:
        msgs.append(Message(role="user", content=prompt))

    return msgs


async def _resolve_adapter(data: dict) -> InferenceAdapter:
    """Lazy-load + return the v2 adapter for the node's model_key."""
    from src.services import workflow_executor as we

    model_key = data.get("model_key") or data.get("model", "")
    if not model_key:
        raise _LLMExecutionError("LLM 节点缺少 model_key")
    if we._model_manager is None:
        raise _LLMExecutionError("ModelManager 未初始化")
    return await we._model_manager.get_loaded_adapter(model_key)


def _build_request(data: dict, inputs: dict, *, stream: bool) -> TextRequest:
    enable_thinking = str(data.get("enable_thinking", "false")).lower() == "true"
    return TextRequest(
        request_id=str(uuid.uuid4()),
        messages=_build_messages(data, inputs),
        model=data.get("model", ""),
        max_tokens=int(data.get("max_tokens", 2048)),
        temperature=float(data.get("temperature", 0.7)),
        stream=stream,
        enable_thinking=enable_thinking,
        api_key=data.get("api_key"),
    )


@register("llm")
class LLMNode:
    """LLM node supporting both streaming (Streamable) and non-streaming (Invokable)."""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        adapter = await _resolve_adapter(data)
        req = _build_request(data, inputs, stream=False)

        t0 = _time.monotonic()
        result = await adapter.infer(req)
        duration_ms = int((_time.monotonic() - t0) * 1000)

        body = result.metadata.get("raw") or {}
        choices = body.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        text = _strip_thinking(text)
        return {"text": text, "usage": body.get("usage"), "duration_ms": duration_ms}

    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token: OnTokenFn,
    ) -> dict:
        adapter = await _resolve_adapter(data)
        req = _build_request(data, inputs, stream=True)

        t0 = _time.monotonic()
        full_text = ""
        usage: dict | None = None
        async for ev in adapter.infer_stream(req):
            assert isinstance(ev, StreamEvent)
            if ev.type == "delta":
                token = ev.payload.get("content", "") or ""
                if token:
                    if on_token is not None:
                        await on_token(token)
                    full_text += token
            elif ev.type == "done":
                usage = ev.payload.get("usage")
            elif ev.type == "error":
                raise _LLMExecutionError(f"LLM stream error: {ev.payload}")

        duration_ms = int((_time.monotonic() - t0) * 1000)
        return {
            "text": _strip_thinking(full_text),
            "usage": usage,
            "duration_ms": duration_ms,
        }
