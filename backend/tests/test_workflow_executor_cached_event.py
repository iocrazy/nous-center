"""PR-6: node_complete event carries cached flag from runner outputs."""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


class _Client:
    async def run_node(self, spec, *, workflow_name=""):
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u", "cached": True}, error=None, duration_ms=3)


@pytest.mark.asyncio
async def test_node_complete_includes_cached():
    events = []
    async def on_prog(e): events.append(e)
    wf = {"nodes": [{"id": "g", "type": "flux2_vae_decode", "data": {"seed": 1}}], "edges": []}
    ex = WorkflowExecutor(wf, on_progress=on_prog, runner_clients={"image": _Client()}, task_id=5)
    await ex.execute()
    complete = [e for e in events if e.get("type") == "node_complete" and e["node_id"] == "g"]
    assert complete and complete[-1].get("cached") is True
