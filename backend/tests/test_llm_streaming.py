"""Tests for LLM streaming support in workflow_executor."""

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
    mock_response.status_code = 200
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
    mock_response.status_code = 200
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


async def test_llm_streaming_dispatch_pushes_node_stream_events():
    """End-to-end: WorkflowExecutor dispatch + LLMNode.stream should emit one
    node_stream event per streamed token (with content=<token>, node_id=<id>),
    and a terminal node_end_streaming carrying the resolved usage dict.

    This replaces the old _exec_llm test: the event shape and responsibility
    boundary moved in W-T4.5 — LLMNode only pumps tokens; the dispatcher wraps
    them as events.
    """
    from src.services import workflow_executor as we
    from src.services.workflow_executor import WorkflowExecutor

    async def fake_stream_llm(base_url, params, on_token=None):
        for token in ["Streaming", " reply"]:
            if on_token:
                await on_token(token)
        we._last_stream_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
        return "Streaming reply"

    events: list[dict] = []

    async def on_progress(event: dict) -> None:
        events.append(event)

    workflow = {
        "nodes": [
            {
                "id": "in",
                "type": "text_input",
                "data": {"text": "hello"},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "llm_node_1",
                "type": "llm",
                "data": {
                    "model": "test",
                    "base_url": "http://localhost:8100",
                    "stream": True,
                },
                "position": {"x": 1, "y": 0},
            },
        ],
        "edges": [
            {"source": "in", "target": "llm_node_1",
             "sourceHandle": "text", "targetHandle": "text"},
        ],
    }

    with patch.object(we, "_stream_llm", fake_stream_llm):
        executor = WorkflowExecutor(workflow, on_progress=on_progress)
        result = await executor.execute()

    assert result["outputs"]["llm_node_1"]["text"] == "Streaming reply"

    stream_events = [e for e in events if e.get("type") == "node_stream"
                     and e.get("node_id") == "llm_node_1"]
    assert len(stream_events) == 2
    assert stream_events[0]["content"] == "Streaming"
    assert stream_events[1]["content"] == " reply"

    end_events = [e for e in events if e.get("type") == "node_end_streaming"]
    assert len(end_events) == 1
    assert end_events[0]["node_id"] == "llm_node_1"
    assert end_events[0]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
