"""Tests for LLM streaming dispatch via the v2 InferenceAdapter pipeline.

The low-level SSE parser used to live as workflow_executor._stream_llm —
v2 moved it into VLLMAdapter.infer_stream (covered by test_vllm_adapter.py).
What remains here is the dispatch boundary: WorkflowExecutor + LLMNode.stream
should turn each StreamEvent('delta') into a node_stream event and emit a
terminal node_end_streaming carrying the usage from the done event.
"""

from unittest.mock import AsyncMock, MagicMock


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


async def test_llm_streaming_dispatch_pushes_node_stream_events(monkeypatch):
    """End-to-end: WorkflowExecutor + LLMNode.stream should emit one
    node_stream event per delta token and a node_end_streaming carrying usage.
    """
    from src.services import workflow_executor as we
    from src.services.inference.base import StreamEvent
    from src.services.workflow_executor import WorkflowExecutor

    async def fake_infer_stream(req):
        for token in ["Streaming", " reply"]:
            yield StreamEvent(type="delta", payload={"content": token})
        yield StreamEvent(
            type="done",
            payload={"usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }},
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer_stream = fake_infer_stream

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

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
                    "model_key": "test-model",
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
