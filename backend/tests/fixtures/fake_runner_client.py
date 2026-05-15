"""Lane S: 纯 Python RunnerClient stub —— 给 WorkflowExecutor 节点分流测试用。

不是 Lane C 的 subprocess fake_runner —— 这个 stub 只实现 executor 直接看到的
`async run_node(node_spec, inputs) -> dict` 接口（spec §3.3 RunNode/NodeResult
节点级 RPC）。Lane C 落地真 RunnerClient 后，executor 代码不变，只换注入对象。
"""
from __future__ import annotations

from typing import Any


class FakeRunnerClient:
    """可配置的 RunnerClient stub。

    - results: node_id -> 该节点 run_node 返回的 outputs dict
    - fail_nodes: 这些 node_id 调 run_node 时抛 RuntimeError
    - calls: 按调用顺序记录 (node_id, node_type, inputs)，给断言用
    """

    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        fail_nodes: set[str] | None = None,
    ):
        self._results = results or {}
        self._fail_nodes = fail_nodes or set()
        self.calls: list[tuple[str, str, dict]] = []

    async def run_node(
        self, node: dict, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """对齐 spec §3.3：主进程投 RunNode，runner 回 NodeResult.outputs。"""
        node_id = node["id"]
        node_type = node["type"]
        self.calls.append((node_id, node_type, dict(inputs)))
        if node_id in self._fail_nodes:
            raise RuntimeError(f"fake runner: node {node_id} failed")
        return self._results.get(node_id, {"result": f"dispatched:{node_id}"})
