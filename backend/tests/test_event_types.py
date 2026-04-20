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
