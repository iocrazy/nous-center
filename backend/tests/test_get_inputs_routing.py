"""_get_inputs 精确路由 — bug hunt round2 #8。"""
from src.services.workflow_executor import WorkflowExecutor


def _ex(edges):
    ex = WorkflowExecutor({"nodes": [], "edges": edges})
    return ex


def test_handled_edges_route_precisely_no_collision():
    """两条带 handle 的入边、上游同名输出键 → 各自路由到不同 target,不互相覆盖、不 spread 污染。"""
    ex = _ex([
        {"source": "a", "sourceHandle": "text", "target": "c", "targetHandle": "x"},
        {"source": "b", "sourceHandle": "text", "target": "c", "targetHandle": "y"},
    ])
    ex._outputs = {
        "a": {"text": "AAA", "extra": "a-extra"},
        "b": {"text": "BBB", "extra": "b-extra"},
    }
    inp = ex._get_inputs("c")
    assert inp["x"] == "AAA" and inp["y"] == "BBB"  # 精确、不混
    assert "extra" not in inp  # 显式 handle 边不再 spread 上游其它键


def test_handleless_edge_keeps_spread_fallback():
    """无 sourceHandle 的老连线仍把上游全部输出灌进来(向后兼容)。"""
    ex = _ex([{"source": "a", "target": "c", "targetHandle": ""}])
    ex._outputs = {"a": {"text": "T", "value": 1}}
    inp = ex._get_inputs("c")
    assert inp["text"] == "T" and inp["value"] == 1  # spread 保留
