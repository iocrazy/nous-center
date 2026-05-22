"""带 seed 的 image dispatch 节点 is_deterministic=True (spec §3.3)。

收敛后用细粒度图终端 flux2_vae_decode 当 dispatch 节点(image_generate 已删)。
is_deterministic 在 _dispatch_node 据 merged_inputs.seed 计算,与节点类型无关。
"""
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
    wf = {"nodes": [{"id": "g", "type": "flux2_vae_decode", "data": node_data}], "edges": []}
    client = _CapturingClient()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=7)
    return ex, client


@pytest.mark.asyncio
async def test_seed_sets_deterministic():
    ex, client = _exec({"seed": 42})
    await ex._dispatch_node(ex._node_map["g"], {"seed": 42})
    assert client.spec.is_deterministic is True


@pytest.mark.asyncio
async def test_no_seed_not_deterministic():
    ex, client = _exec({})
    await ex._dispatch_node(ex._node_map["g"], {})
    assert client.spec.is_deterministic is False


@pytest.mark.asyncio
async def test_empty_seed_not_deterministic():
    ex, client = _exec({"seed": ""})
    await ex._dispatch_node(ex._node_map["g"], {"seed": ""})
    assert client.spec.is_deterministic is False
