"""Lane C-T1 · Ollama ↔ OpenAI translation unit tests.

The adapter is pure: input → output, no IO, no DB. Each function here
gets nailed down with a shape test so a wire-format change is caught
before it hits the user.
"""

from __future__ import annotations

import json

import pytest

from src.services.ollama_adapter import (
    ollama_chat_to_openai,
    openai_chat_to_ollama,
    openai_sse_chunk_to_ollama_ndjson,
    ollama_generate_to_openai,
    openai_chat_to_ollama_generate,
)


# ---------- /api/chat request translation ----------


def test_chat_request_basic():
    ollama = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    openai = ollama_chat_to_openai(ollama)
    assert openai["messages"] == [{"role": "user", "content": "hi"}]
    assert openai["stream"] is True
    # Ollama `model` is preserved for the resolver; vLLM proxy will overwrite.
    assert openai["model"] == "qwen3.5"


def test_chat_request_options_map_to_openai_fields():
    """Ollama uses {"options": {"temperature": .., "num_predict": ..}}."""
    ollama = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": "hi"}],
        "options": {
            "temperature": 0.7,
            "num_predict": 100,
            "top_p": 0.9,
            "seed": 42,
        },
    }
    openai = ollama_chat_to_openai(ollama)
    assert openai["temperature"] == 0.7
    assert openai["max_tokens"] == 100
    assert openai["top_p"] == 0.9
    assert openai["seed"] == 42
    assert "options" not in openai


def test_chat_request_stream_defaults_true():
    """Ollama defaults stream to True; OpenAI clients default False."""
    ollama = {"model": "m", "messages": []}
    openai = ollama_chat_to_openai(ollama)
    assert openai["stream"] is True


# ---------- /api/chat response translation (non-stream) ----------


def test_chat_response_non_stream():
    openai_resp = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "qwen3.5",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    ollama = openai_chat_to_ollama(openai_resp, model_name="qwen3.5")
    assert ollama["model"] == "qwen3.5"
    assert ollama["message"] == {"role": "assistant", "content": "Hello"}
    assert ollama["done"] is True
    assert ollama["done_reason"] == "stop"
    assert ollama["prompt_eval_count"] == 10
    assert ollama["eval_count"] == 5
    assert "created_at" in ollama


def test_chat_response_missing_usage_still_valid():
    openai_resp = {
        "choices": [{
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
    }
    ollama = openai_chat_to_ollama(openai_resp, model_name="x")
    assert ollama["done"] is True
    assert ollama["message"]["content"] == "ok"


# ---------- /api/chat streaming (SSE chunk → NDJSON line) ----------


def test_sse_content_chunk_to_ndjson():
    sse = 'data: {"choices":[{"delta":{"content":"He"},"index":0}]}'
    line = openai_sse_chunk_to_ollama_ndjson(sse, model_name="qwen3.5")
    obj = json.loads(line)
    assert obj["model"] == "qwen3.5"
    assert obj["message"] == {"role": "assistant", "content": "He"}
    assert obj["done"] is False


def test_sse_final_chunk_with_finish_reason_to_ndjson_done():
    sse = (
        'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
    )
    line = openai_sse_chunk_to_ollama_ndjson(sse, model_name="qwen3.5")
    obj = json.loads(line)
    assert obj["done"] is True
    assert obj["done_reason"] == "stop"
    assert obj["prompt_eval_count"] == 10
    assert obj["eval_count"] == 5


def test_sse_done_marker_becomes_none():
    """`data: [DONE]` is an SSE protocol marker. We've already emitted the
    terminal chunk (with finish_reason); nothing left to send."""
    assert openai_sse_chunk_to_ollama_ndjson(
        "data: [DONE]", model_name="qwen3.5",
    ) is None


def test_sse_empty_line_returns_none():
    assert openai_sse_chunk_to_ollama_ndjson("", model_name="x") is None


def test_sse_role_only_chunk_skipped():
    """First chunk from vLLM usually has only role=assistant; no content
    to emit yet."""
    sse = 'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}'
    assert openai_sse_chunk_to_ollama_ndjson(sse, model_name="x") is None


# ---------- /api/generate request/response ----------


def test_generate_request_to_openai_chat():
    """Ollama generate is single-prompt; we fake it as a one-message chat."""
    ollama = {"model": "qwen3.5", "prompt": "once upon a time", "stream": True}
    openai = ollama_generate_to_openai(ollama)
    assert openai["model"] == "qwen3.5"
    assert openai["messages"] == [
        {"role": "user", "content": "once upon a time"},
    ]
    assert openai["stream"] is True


def test_generate_response_non_stream_shape():
    openai_resp = {
        "choices": [{
            "message": {"role": "assistant", "content": "world"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    ollama = openai_chat_to_ollama_generate(openai_resp, model_name="qwen3.5")
    assert ollama["response"] == "world"
    assert ollama["done"] is True
    assert ollama["prompt_eval_count"] == 3
    assert ollama["eval_count"] == 1
