"""Lane S: 纯 Python RunnerClient stub —— 给 WorkflowExecutor 节点分流测试用。

签名对齐真 Lane C RunnerClient.run_node(spec: P.RunNode, *, on_progress=None,
workflow_name=""),executor 改完构造 P.RunNode 后 stub 跟着改 —— 不再 (node, inputs)。
"""
from __future__ import annotations

from typing import Any


class FakeRunnerClient:
    """可配置的 RunnerClient stub,签名匹配真 Lane C RunnerClient。

    - results: node_id -> 该节点 run_node 返回的 outputs dict
    - fail_nodes: 这些 node_id 调 run_node 时抛 RuntimeError
    - calls: 按调用顺序记录 (node_id, node_type, inputs, workflow_name)
    """

    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        fail_nodes: set[str] | None = None,
    ):
        self._results = results or {}
        self._fail_nodes = fail_nodes or set()
        self.calls: list[tuple[str, str, dict, str]] = []
        # Lane K follow-up: 模拟真 RunnerClient.current_dispatch 接口供测试用
        self.current_dispatch: dict | None = None

    async def run_node(
        self,
        spec: Any,                  # P.RunNode dataclass
        *,
        on_progress: Any = None,    # 真客户端的 callable;stub 不调
        workflow_name: str = "",
    ) -> Any:
        """对齐 spec §3.3 + Lane C client.run_node 签名。返回类 P.NodeResult dict。"""
        node_id = spec.node_id
        node_type = spec.node_type
        self.calls.append((node_id, node_type, dict(spec.inputs), workflow_name))
        if node_id in self._fail_nodes:
            raise RuntimeError(f"fake runner: node {node_id} failed")
        outputs = self._results.get(node_id, {"result": f"dispatched:{node_id}"})
        # WorkflowExecutor._dispatch_node 当前直接拿到 client.run_node 的返回值,
        # 老 stub 返回 outputs dict。保持兼容 —— 真 RunnerClient 返回 P.NodeResult,
        # 但 executor 也只读 .outputs;两边各自跑通,真接通后统一(见 Lane K
        # follow-up issue:executor _dispatch_node 应 return result.outputs)。
        return outputs
