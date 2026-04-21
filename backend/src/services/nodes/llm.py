"""LLM node: implements InvokableNode (non-stream) + StreamableNode (stream).

Migrated from workflow_executor._exec_llm as part of Wave 1 Task 4.3. Key change:
on_token is passed explicitly to stream(); no more global progress-callback ref.

The stream path reuses workflow_executor._stream_llm (the low-level httpx SSE
helper established in W-T1.2) and reads workflow_executor._last_stream_usage
for the final usage dict. This preserves the exact token-stats behavior the
frontend depends on, while the node itself stays clean of global progress refs.
"""

from __future__ import annotations

import logging
import time as _time
import urllib.parse
from typing import Any

import httpx

from src.services.nodes.base import OnTokenFn
from src.services.nodes.registry import register
from src.utils.constants import ALLOWED_LLM_HOSTS

logger = logging.getLogger(__name__)


class _LLMExecutionError(Exception):
    """Internal error raised by LLMNode. WorkflowExecutor wraps these."""


def _validate_llm_url(url: str) -> str:
    """Ensure LLM base_url only points to localhost."""
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and parsed.hostname not in ALLOWED_LLM_HOSTS:
        raise _LLMExecutionError(
            f"LLM base_url 只允许 localhost，收到: {parsed.hostname}"
        )
    return url


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output, return only the final answer."""
    import re
    result = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    return result.strip()


async def _resolve_base_url_and_adapter(data: dict) -> tuple[str, Any]:
    """Resolve base_url + (optional) adapter from data.model_key via ModelManager.

    Mirrors the logic at the top of the old _exec_llm.
    """
    # Lazy import to avoid circular deps.
    from src.services import workflow_executor as we

    model_key = data.get("model_key", "")
    base_url = data.get("base_url", "")
    adapter = None

    if model_key and we._model_manager is not None:
        adapter = we._model_manager.get_adapter(model_key)
        if adapter is None or not getattr(adapter, "is_loaded", False):
            await we._model_manager.load_model(model_key)
            adapter = we._model_manager.get_adapter(model_key)
        if adapter is not None and hasattr(adapter, "base_url"):
            base_url = adapter.base_url

    if not base_url:
        from src.config import get_settings
        base_url = get_settings().VLLM_BASE_URL

    _validate_llm_url(base_url)
    return base_url, adapter


def _build_messages(data: dict, inputs: dict) -> list[dict]:
    """Build OpenAI-format messages from inputs.

    Priority:
    1. If inputs['messages'] is a list, use it directly (Wave 1 contract).
    2. Else build from data['system'] + inputs['prompt'/'text'] + multimodal
       (images/audio), matching the legacy _exec_llm behavior.
    """
    supplied = inputs.get("messages")
    if isinstance(supplied, list) and supplied:
        return supplied

    prompt = inputs.get("prompt") or inputs.get("text", "")
    if not prompt:
        raise _LLMExecutionError("LLM 节点缺少 prompt 输入")

    messages: list[dict] = []
    system_msg = data.get("system")
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    # Multimodal: images + audio
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
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


async def _clamp_max_tokens(data: dict, base_url: str, adapter: Any) -> int:
    """Clamp max_tokens to the model's advertised max_model_len.

    Checks adapter first (no HTTP), then falls back to /v1/models GET.
    """
    max_tokens = int(data.get("max_tokens", 2048))
    model_max = 4096  # safe default

    if adapter is not None:
        model_max = getattr(adapter, "max_model_len", model_max) or model_max
    else:
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
    return max_tokens


@register("llm")
class LLMNode:
    """LLM node supporting both streaming (Streamable) and non-streaming (Invokable)
    execution paths.

    Unlike the legacy _exec_llm, this class never touches any module-level
    progress-callback global. Streaming callers pass an explicit on_token coroutine.
    """

    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Non-streaming LLM call. Returns {text, usage, duration_ms}."""
        base_url, adapter = await _resolve_base_url_and_adapter(data)
        messages = _build_messages(data, inputs)

        enable_thinking = str(data.get("enable_thinking", "false")).lower() == "true"
        max_tokens = await _clamp_max_tokens(data, base_url, adapter)

        body: dict[str, Any] = {
            "model": data.get("model", ""),
            "messages": messages,
            "temperature": data.get("temperature", 0.7),
            "max_tokens": max_tokens,
            # Always pass explicit value — Qwen3's chat template defaults to
            # thinking=True, so omitting the flag when UI picks 关闭 still
            # produces reasoning traces in the output.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

        headers: dict[str, str] = {}
        api_key = data.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = _time.monotonic()
        async with httpx.AsyncClient(timeout=300, proxy=None) as _client:
            resp = await _client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            if resp.status_code != 200:
                try:
                    err = resp.json()
                    detail = err.get("error", {}).get("message", resp.text[:300])
                except Exception:
                    detail = resp.text[:300]
                raise _LLMExecutionError(
                    f"LLM API error ({resp.status_code}): {detail}"
                )
            resp_data = resp.json()
            result = resp_data["choices"][0]["message"]["content"]

        duration_ms = int((_time.monotonic() - t0) * 1000)
        result = _strip_thinking(result)
        usage = resp_data.get("usage")
        return {"text": result, "usage": usage, "duration_ms": duration_ms}

    async def stream(
        self,
        data: dict,
        inputs: dict,
        on_token: OnTokenFn,
    ) -> dict:
        """Streaming LLM call. Invokes on_token per chunk. Returns {text, usage, duration_ms}.

        NOTE: this does not emit node_end_streaming — that is the dispatcher's
        job in Subtask 4.5. This method is only responsible for pumping tokens
        through on_token and returning the final dict.
        """
        # Reuse the low-level SSE helper from workflow_executor. Captures usage
        # into workflow_executor._last_stream_usage via include_usage chunk.
        from src.services import workflow_executor as we

        base_url, adapter = await _resolve_base_url_and_adapter(data)
        messages = _build_messages(data, inputs)

        enable_thinking = str(data.get("enable_thinking", "false")).lower() == "true"
        max_tokens = await _clamp_max_tokens(data, base_url, adapter)

        params: dict[str, Any] = {
            "model": data.get("model", ""),
            "messages": messages,
            "temperature": data.get("temperature", 0.7),
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            # Always explicit — see invoke() comment.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

        t0 = _time.monotonic()
        # Pass on_token through directly — the helper already awaits per chunk.
        full_text = await we._stream_llm(base_url, params, on_token=on_token)
        duration_ms = int((_time.monotonic() - t0) * 1000)
        full_text = _strip_thinking(full_text)

        return {
            "text": full_text,
            "usage": we._last_stream_usage,
            "duration_ms": duration_ms,
        }
