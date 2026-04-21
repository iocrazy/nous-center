"""Ollama ↔ OpenAI wire-format translation.

Pure functions. The HTTP layer owns auth, resolution, and the actual
vLLM call; this module just reshapes JSON.

Two directions:
  ollama_chat_to_openai(body)        — /api/chat request body
  openai_chat_to_ollama(resp, name)  — /api/chat non-stream response
  openai_sse_chunk_to_ollama_ndjson  — /api/chat streaming, per-line
  ollama_generate_to_openai(body)    — /api/generate request body
  openai_chat_to_ollama_generate     — /api/generate response

Design notes:
- Ollama defaults `stream=true`; OpenAI defaults `false`. We respect
  the client's intent and default to True only when unset.
- Ollama's `options` map flattens to OpenAI top-level: `temperature`,
  `max_tokens` (from `num_predict`), `top_p`, `seed`.
- vLLM's first SSE chunk has `delta.role` and no content — we skip it.
  The final chunk has `finish_reason` set and `usage` populated; that's
  when we emit the terminal NDJSON line with `done=true`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# --- option key mapping (Ollama → OpenAI) ---
_OPTION_TO_OPENAI_KEY = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    # Ollama's `num_predict` is max output tokens; OpenAI uses `max_tokens`.
    "num_predict": "max_tokens",
    # `num_ctx` is context window — caller-side concern, not a sampling param.
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def ollama_chat_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate /api/chat → OpenAI /v1/chat/completions body shape."""
    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": body.get("messages", []),
        # Ollama defaults stream to True.
        "stream": bool(body.get("stream", True)),
    }

    options = body.get("options") or {}
    for ollama_key, openai_key in _OPTION_TO_OPENAI_KEY.items():
        if ollama_key in options:
            out[openai_key] = options[ollama_key]

    return out


def ollama_generate_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate /api/generate → OpenAI /v1/chat/completions body shape.

    Ollama's generate is a single-prompt API; we wrap the prompt as a
    user message. If `system` is set, prepend it as a system message.
    """
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append({"role": "user", "content": body.get("prompt", "")})

    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": bool(body.get("stream", True)),
    }

    options = body.get("options") or {}
    for ollama_key, openai_key in _OPTION_TO_OPENAI_KEY.items():
        if ollama_key in options:
            out[openai_key] = options[ollama_key]
    return out


def openai_chat_to_ollama(
    resp: dict[str, Any], *, model_name: str,
) -> dict[str, Any]:
    """Translate a non-stream /v1/chat/completions response → /api/chat."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {"role": "assistant", "content": ""}
    usage = resp.get("usage") or {}
    return {
        "model": model_name,
        "created_at": _iso_now(),
        "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        },
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def openai_chat_to_ollama_generate(
    resp: dict[str, Any], *, model_name: str,
) -> dict[str, Any]:
    """Translate a non-stream /v1/chat/completions response → /api/generate.

    /api/generate packs the assistant text into `response` (string), not
    into a message object — matches Ollama's flat shape.
    """
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {"content": ""}
    usage = resp.get("usage") or {}
    return {
        "model": model_name,
        "created_at": _iso_now(),
        "response": message.get("content", ""),
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def openai_sse_chunk_to_ollama_ndjson(
    sse_line: str, *, model_name: str,
) -> str | None:
    """Translate one SSE line to one NDJSON line, or None to skip.

    Skip signals (return None):
      - empty line (SSE keepalive)
      - `data: [DONE]` (caller's terminator; we already emitted terminal done=true)
      - first chunk with only `delta.role` (no content to forward)
      - malformed JSON (keep the stream alive, drop the chunk)

    Terminal chunk (`finish_reason` set) → NDJSON with `done: true`.
    Content chunk (`delta.content` present) → NDJSON with `done: false`.
    """
    line = sse_line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]" or not payload:
        return None

    try:
        chunk = json.loads(payload)
    except Exception:
        return None

    choices = chunk.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    finish_reason = choices[0].get("finish_reason")
    content = delta.get("content")

    if finish_reason is not None:
        usage = chunk.get("usage") or {}
        return json.dumps({
            "model": model_name,
            "created_at": _iso_now(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": finish_reason,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        })

    if content is None or content == "":
        # Role-only first chunk, or empty content keepalive.
        return None

    return json.dumps({
        "model": model_name,
        "created_at": _iso_now(),
        "message": {"role": "assistant", "content": content},
        "done": False,
    })
