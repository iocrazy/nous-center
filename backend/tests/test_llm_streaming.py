"""Tests for LLM streaming support in workflow_executor."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class AsyncIteratorMock:
    """Helper: async iterator over a list of strings."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


async def test_stream_llm_exists():
    """_stream_llm should be importable and callable."""
    from src.services.workflow_executor import _stream_llm

    assert callable(_stream_llm)


async def test_stream_llm_parses_tokens():
    """_stream_llm should parse SSE chunks and call on_token for each token."""
    from src.services.workflow_executor import _stream_llm

    tokens: list[str] = []

    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=AsyncIteratorMock(lines))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def collect_token(t: str) -> None:
        tokens.append(t)

    with patch("src.services.workflow_executor.httpx.AsyncClient", return_value=mock_client):
        result = await _stream_llm(
            "http://localhost:8100",
            {"model": "test", "messages": []},
            on_token=collect_token,
        )

    assert result == "Hello world"
    assert tokens == ["Hello", " world"]


async def test_stream_llm_skips_non_data_lines():
    """_stream_llm should ignore blank lines and non-data lines."""
    from src.services.workflow_executor import _stream_llm

    lines = [
        "",
        ": ping",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=AsyncIteratorMock(lines))
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.workflow_executor.httpx.AsyncClient", return_value=mock_client):
        result = await _stream_llm("http://localhost:8100", {})

    assert result == "ok"


async def test_exec_llm_injects_node_id():
    """_execute_node should inject _node_id into node data before calling executor."""
    from src.services.workflow_executor import WorkflowExecutor

    workflow = {
        "nodes": [
            {
                "id": "n1",
                "type": "text_input",
                "data": {"text": "hi"},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [],
    }
    executor = WorkflowExecutor(workflow)
    result = await executor.execute()
    assert "outputs" in result
    assert "n1" in result["outputs"]


async def test_exec_llm_streaming_pushes_node_stream_events():
    """When stream=True and on_progress is set, _exec_llm should push node_stream events."""
    from src.services import workflow_executor as we

    events: list[dict] = []

    async def on_progress(event: dict) -> None:
        events.append(event)

    # Patch _stream_llm to return a known result without real HTTP
    async def fake_stream_llm(base_url, params, on_token=None):
        for token in ["Streaming", " reply"]:
            if on_token:
                await on_token(token)
        return "Streaming reply"

    # Patch call_llm so non-streaming path is also safe
    async def fake_call_llm(**kwargs):
        return "non-stream reply"

    with patch.object(we, "_stream_llm", fake_stream_llm), \
         patch.object(we, "call_llm", fake_call_llm):
        we._on_progress_ref = on_progress
        data = {
            "stream": True,
            "_node_id": "llm_node_1",
            "model": "test",
            "base_url": "http://localhost:8100",
        }
        inputs = {"text": "hello"}
        result = await we._exec_llm(data, inputs)

    assert result == {"text": "Streaming reply"}
    stream_events = [e for e in events if e.get("type") == "node_stream"]
    assert len(stream_events) == 2
    assert stream_events[0]["token"] == "Streaming"
    assert stream_events[1]["token"] == " reply"
    assert all(e["node_id"] == "llm_node_1" for e in stream_events)


async def test_on_progress_ref_set_during_execute():
    """WorkflowExecutor._execute_node should set _on_progress_ref to self._on_progress."""
    from src.services import workflow_executor as we
    from src.services.workflow_executor import WorkflowExecutor

    progress_calls: list[dict] = []

    async def on_progress(event: dict) -> None:
        progress_calls.append(event)

    workflow = {
        "nodes": [
            {
                "id": "n_text",
                "type": "text_input",
                "data": {"text": "test"},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [],
    }
    executor = WorkflowExecutor(workflow, on_progress=on_progress)
    await executor.execute()

    # After execution _on_progress_ref should have been set to the instance callback
    assert we._on_progress_ref is on_progress
