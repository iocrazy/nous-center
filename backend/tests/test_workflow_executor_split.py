"""Lane S: WorkflowExecutor 节点分流执行测试。"""
import pytest

from src.services.workflow_executor import ExecutionError, WorkflowExecutor
from tests.fixtures.fake_runner_client import FakeRunnerClient


def _wf(nodes, edges=None):
    return {"nodes": nodes, "edges": edges or []}


def _collector():
    events: list[dict] = []

    async def _on(e: dict) -> None:
        events.append(e)

    return events, _on


@pytest.mark.asyncio
async def test_fake_runner_client_records_calls():
    """stub self-check：run_node 记录调用、按 node_id 返回配置结果。"""
    rc = FakeRunnerClient(results={"n1": {"image_url": "x.png"}})
    out = await rc.run_node({"id": "n1", "type": "image_generate"}, {"prompt": "cat"})
    assert out == {"image_url": "x.png"}
    assert rc.calls == [("n1", "image_generate", {"prompt": "cat"})]


@pytest.mark.asyncio
async def test_fake_runner_client_fail_nodes():
    rc = FakeRunnerClient(fail_nodes={"bad"})
    with pytest.raises(RuntimeError, match="node bad failed"):
        await rc.run_node({"id": "bad", "type": "image_generate"}, {})


@pytest.mark.asyncio
async def test_inline_node_does_not_touch_runner_client():
    """纯 inline workflow：runner_client 一次都不该被调。"""
    rc = FakeRunnerClient()
    wf = _wf([{"id": "t1", "type": "text_input", "data": {"text": "hello"}}])
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    assert rc.calls == []
    assert "t1" in result["outputs"]


@pytest.mark.asyncio
async def test_dispatch_node_routes_to_runner_client():
    """image_generate 节点 → RunnerClient.run_node，结果进 outputs。"""
    rc = FakeRunnerClient(results={"img": {"image_url": "out.png"}})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {"prompt": "cat"}}])
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    assert rc.calls[0][0] == "img"
    assert result["outputs"]["img"] == {"image_url": "out.png"}


@pytest.mark.asyncio
async def test_mixed_workflow_inline_then_dispatch():
    """text_input(inline) → image_generate(dispatch)：上游 inline 输出进下游 dispatch 的 inputs。"""
    rc = FakeRunnerClient(results={"img": {"image_url": "out.png"}})
    wf = _wf(
        nodes=[
            {"id": "t", "type": "text_input", "data": {"text": "a cat"}},
            {"id": "img", "type": "image_generate", "data": {}},
        ],
        edges=[{"source": "t", "target": "img",
                "sourceHandle": "text", "targetHandle": "prompt"}],
    )
    ex = WorkflowExecutor(wf, runner_client=rc)
    result = await ex.execute()
    # dispatch 节点拿到了 inline 上游的输出
    assert "text" in rc.calls[0][2] or "prompt" in rc.calls[0][2]
    assert result["outputs"]["img"] == {"image_url": "out.png"}


@pytest.mark.asyncio
async def test_dispatch_node_without_runner_client_raises():
    """runner_client=None 但 workflow 含 dispatch 节点 → ExecutionError（不静默 inline 跑 GPU 节点）。"""
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=None)
    with pytest.raises(ExecutionError, match="runner"):
        await ex.execute()


@pytest.mark.asyncio
async def test_dispatch_node_failure_wrapped():
    """runner 抛错 → ExecutionError，node_error progress 事件发出。"""
    events, on_progress = _collector()
    rc = FakeRunnerClient(fail_nodes={"img"})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=rc, on_progress=on_progress)
    with pytest.raises(ExecutionError):
        await ex.execute()
    assert any(e["type"] == "node_error" and e["node_id"] == "img" for e in events)


@pytest.mark.asyncio
async def test_progress_events_unchanged_for_dispatch():
    """dispatch 节点同样发 node_start / node_complete progress 事件。"""
    events, on_progress = _collector()
    rc = FakeRunnerClient(results={"img": {"image_url": "x"}})
    wf = _wf([{"id": "img", "type": "image_generate", "data": {}}])
    ex = WorkflowExecutor(wf, runner_client=rc, on_progress=on_progress)
    await ex.execute()
    types = [e["type"] for e in events]
    assert "node_start" in types and "node_complete" in types
