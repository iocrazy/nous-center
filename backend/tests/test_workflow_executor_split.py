"""Lane S: WorkflowExecutor 节点分流执行测试。"""
import pytest

from tests.fixtures.fake_runner_client import FakeRunnerClient


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
