import pytest

from src.services.workflow_executor import EVENT_TYPES


def test_all_new_event_types_defined():
    expected_new = {
        "node_end_streaming",
        "workflow_interrupt",
        "workflow_resume",
        "function_call",
        "tool_response",
        "tool_streaming_response",
    }
    assert expected_new.issubset(set(EVENT_TYPES))


def test_existing_event_types_preserved():
    """Ensure no regression on existing events."""
    expected_existing = {"node_start", "node_stream", "node_complete", "node_error", "complete"}
    assert expected_existing.issubset(set(EVENT_TYPES))


@pytest.mark.asyncio
async def test_llm_stream_emits_node_end_streaming(mock_llm_stream):
    """WorkflowExecutor dispatch emits node_end_streaming after LLMNode.stream()
    returns, carrying the final usage dict that the streaming helper captured.

    This contract is load-bearing for the frontend token-stats UI and for Wave 1
    coze-style event consumers. Dispatch emits the event — LLMNode.stream does
    not; see workflow_executor._execute_node.
    """
    from src.services.workflow_executor import WorkflowExecutor

    events: list[dict] = []

    async def on_progress(ev: dict) -> None:
        events.append(ev)

    # LLMNode reads inputs.prompt or inputs.text. Feed it via a text_input node.
    workflow = {
        "nodes": [
            {
                "id": "in",
                "type": "text_input",
                "data": {"text": "hi"},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "llm-1",
                "type": "llm",
                "data": {
                    "model": "qwen3.5",
                    "base_url": "http://localhost:8100",
                    "stream": True,
                    "max_tokens": 128,
                },
                "position": {"x": 1, "y": 0},
            },
        ],
        "edges": [
            {"source": "in", "target": "llm-1",
             "sourceHandle": "text", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(workflow, on_progress=on_progress)
    await executor.execute()

    event_types = [e["type"] for e in events]
    assert "node_stream" in event_types, f"expected node_stream events, got {event_types}"
    assert "node_end_streaming" in event_types, f"expected node_end_streaming, got {event_types}"

    idx_stream = next(i for i, t in enumerate(event_types) if t == "node_stream")
    idx_end = event_types.index("node_end_streaming")
    assert idx_stream < idx_end

    end_ev = next(e for e in events if e["type"] == "node_end_streaming")
    assert end_ev["node_id"] == "llm-1"
    assert end_ev["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }


def test_ws_handler_is_passthrough():
    """ws handler forwards event dicts verbatim (no type-based filter).

    Confirmed by reading:
      - backend/src/api/routes/workflows.py :: execute_workflow_direct
        -> `broadcast_progress(event)` loops `_ws_connections[channel_id]`
        and calls `ws.send_json(event)` unconditionally.
      - backend/src/api/routes/instance_service.py :: broadcast_progress
        -> same pattern, `ws.send_json(data)` with no type check.
      - backend/src/api/main.py :: workflow_progress_ws
        -> endpoint only accepts connection and stores the WebSocket,
        performs no event-level filtering.

    If anyone ever adds a `if event["type"] in {...}` whitelist, this test
    must fail so the whitelist is forced to cover the full EVENT_TYPES set.
    """
    import inspect

    from src.services.workflow_executor import EVENT_TYPES  # noqa: F401  (documents intent)
    from src.api import main as api_main
    from src.api.routes import instance_service, workflows

    for module in (workflows, instance_service, api_main):
        src = inspect.getsource(module)
        assert 'event["type"] in' not in src, (
            f"{module.__name__} has a ws type whitelist on event['type']; "
            "replace the literal set with EVENT_TYPES from workflow_executor."
        )
        assert "event.get('type') in " not in src, (
            f"{module.__name__} has a ws type whitelist on event.get('type'); "
            "replace the literal set with EVENT_TYPES from workflow_executor."
        )
        assert 'event.get("type") in ' not in src, (
            f"{module.__name__} has a ws type whitelist on event.get(\"type\"); "
            "replace the literal set with EVENT_TYPES from workflow_executor."
        )
