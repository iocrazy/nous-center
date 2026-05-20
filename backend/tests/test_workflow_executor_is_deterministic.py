"""PR-4: 带 seed 的 image 节点 dispatch 时 is_deterministic=True (spec §3.3)。"""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


class _CapturingClient:
    def __init__(self):
        self.spec: P.RunNode | None = None

    async def run_node(self, spec, *, workflow_name=""):
        self.spec = spec
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u"}, error=None, duration_ms=1)


def _exec(node_data):
    wf = {"nodes": [{"id": "g", "type": "image_generate", "data": node_data}], "edges": []}
    client = _CapturingClient()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=7)
    return ex, client


@pytest.mark.asyncio
async def test_seed_sets_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x", "seed": 42})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x", "seed": 42})
    assert client.spec.is_deterministic is True


@pytest.mark.asyncio
async def test_no_seed_not_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x"})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x"})
    assert client.spec.is_deterministic is False


@pytest.mark.asyncio
async def test_empty_seed_not_deterministic():
    ex, client = _exec({"model_key": "flux2-klein-9b", "prompt": "x", "seed": ""})
    await ex._dispatch_node(ex._node_map["g"], {"prompt": "x", "seed": ""})
    assert client.spec.is_deterministic is False
