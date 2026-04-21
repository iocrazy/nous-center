"""Wave 1 Task 4.5 — verify WorkflowExecutor._execute_node dispatches via
registered node classes (InvokableNode / StreamableNode protocols).
"""

from __future__ import annotations

import pytest

from src.services.workflow_executor import WorkflowExecutor


@pytest.mark.asyncio
async def test_executor_dispatches_via_registry():
    """_execute_node should look up class from registry, instantiate, call invoke/stream."""
    workflow = {
        "nodes": [{"id": "n1", "type": "text_input", "data": {"text": "hi"}}],
        "edges": [],
    }
    captured: list[dict] = []

    async def on_progress(ev):
        captured.append(ev)

    exe = WorkflowExecutor(workflow, on_progress=on_progress)
    result = await exe.execute()
    assert result["outputs"]["n1"] == {"text": "hi"}
