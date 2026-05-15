"""Lane J test infrastructure: mock vLLM HTTP endpoint (spec §5.6).

Used by LLM-direct-path integration tests — compat routes (openai /
anthropic / ollama / responses) and workflow llm nodes both talk to
vLLM HTTP directly (spec D6/D8). Runs a threaded uvicorn server
exposing /v1/chat/completions (with streaming), /health, and /v1/abort.
Records in-flight concurrency so the "LLM runner is not serialized"
integration case can assert observed parallelism.

Supersedes Lane E's inlined fallback mock.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


class FakeVLLMState:
    """Cross-request shared counters — exposed to tests for assertions."""

    def __init__(self) -> None:
        self.base_url: str = ""
        self.request_count: int = 0
        self.in_flight: int = 0
        self.max_concurrent_seen: int = 0
        self.aborted_ids: list[str] = []
        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self.request_count += 1
            self.in_flight += 1
            if self.in_flight > self.max_concurrent_seen:
                self.max_concurrent_seen = self.in_flight

    def _exit(self) -> None:
        with self._lock:
            self.in_flight -= 1


def _build_app(state: FakeVLLMState) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/abort")
    async def abort(req: Request):
        body = await req.json()
        state.aborted_ids.append(body.get("request_id", ""))
        return {"aborted": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state._enter()
        try:
            # Brief delay so concurrent requests genuinely overlap (so
            # max_concurrent_seen exceeds 1 when callers race).
            await asyncio.sleep(0.05)
            if body.get("stream"):
                async def _gen():
                    for tok in ("hel", "lo"):
                        yield (
                            'data: {"choices":[{"delta":{"content":"'
                            + tok
                            + '"}}]}\n\n'
                        )
                    yield (
                        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(_gen(), media_type="text/event-stream")
            return {
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        finally:
            state._exit()

    return app


class _ThreadedServer:
    """uvicorn running on a daemon thread with port=0 (OS picks free port)."""

    def __init__(self, app: FastAPI) -> None:
        self._config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning"
        )
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> str:
        self._thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            if self._server.started and self._server.servers:
                sock = self._server.servers[0].sockets[0]
                return f"http://127.0.0.1:{sock.getsockname()[1]}"
            time.sleep(0.02)
        raise RuntimeError("fake_vllm server failed to start in 10s")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


@pytest.fixture
def fake_vllm():
    """Start a mock vLLM HTTP server (daemon thread); yield FakeVLLMState
    (which has .base_url + concurrency counters). Teardown shuts the server.
    """
    state = FakeVLLMState()
    server = _ThreadedServer(_build_app(state))
    state.base_url = server.start()
    try:
        yield state
    finally:
        with contextlib.suppress(Exception):
            server.stop()
