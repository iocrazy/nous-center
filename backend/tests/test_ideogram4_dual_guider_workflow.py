"""集成:ideogram4_dual_guider 合并节点在**真 WorkflowExecutor 图执行**里的边路由(发布工作流/外部 API 路径)。

#517 单测直接调 exec_ideogram4_dual_guider;本测补「在真图里两条 MODEL 边(model + unconditional_model
两个 handle)经 _get_inputs 正确路由到合并节点」—— 发布工作流(run_published_workflow)/外部 API 走的就是
WorkflowExecutor 这条图执行 + get_all_executors 解析 inline 节点。不需 GPU(全 inline 描述符)。
"""
from __future__ import annotations

import pytest

from src.services.workflow_executor import WorkflowExecutor


@pytest.fixture(autouse=True)
def _load_node_packages():
    """填充 inline 节点执行器注册表(flux2-components 含 ideogram4_dual_guider)—— 真图执行经
    get_all_executors 解析,测试环境须先 scan_packages 填表。"""
    from nodes import scan_packages
    scan_packages()


def _dual_dit_workflow():
    """两个 Load Diffusion Model(ideogram4 cond/uncond)→ Ideogram-4 双 DiT 合并。"""
    return {
        "nodes": [
            {"id": "cond", "type": "flux2_load_diffusion_model",
             "data": {"file": "/m/ideogram4_fp8_scaled.safe", "adapter_arch": "ideogram4",
                      "weight_dtype": "fp8_e4m3", "device": "cuda:2", "offload": "cpu"},
             "position": {"x": 0, "y": 0}},
            {"id": "uncond", "type": "flux2_load_diffusion_model",
             "data": {"file": "/m/ideogram4_unconditional_fp8_scaled.safe", "adapter_arch": "ideogram4",
                      "weight_dtype": "fp8_e4m3", "device": "cuda:2", "offload": "cpu"},
             "position": {"x": 0, "y": 200}},
            {"id": "guider", "type": "ideogram4_dual_guider", "data": {}, "position": {"x": 300, "y": 100}},
        ],
        "edges": [
            {"id": "e1", "source": "cond", "sourceHandle": "model", "target": "guider", "targetHandle": "model"},
            {"id": "e2", "source": "uncond", "sourceHandle": "model", "target": "guider",
             "targetHandle": "unconditional_model"},
        ],
    }


@pytest.mark.asyncio
async def test_dual_guider_wires_two_models_in_executor():
    """真图执行:两条 MODEL 边经 _get_inputs 路由到合并节点的 model / unconditional_model 两个 handle
    → 合并出带 unconditional_file 的 MODEL(发布工作流/外部 API 同此路径)。"""
    wf = _dual_dit_workflow()
    ex = WorkflowExecutor(wf, runner_client=None)
    node_map = {n["id"]: n for n in wf["nodes"]}

    # 跑两个 loader(inline)→ 存进 _outputs(模拟图执行,经真 get_all_executors → exec_load_diffusion_model)。
    for nid in ("cond", "uncond"):
        out = await ex._execute_inline_node(node_map[nid], {})
        ex._outputs[nid] = out

    # 合并节点:经真 _get_inputs(边路由)拿到两个 MODEL,再经真 exec_ideogram4_dual_guider 合并。
    guider_inputs = ex._get_inputs("guider")
    assert "model" in guider_inputs and "unconditional_model" in guider_inputs, \
        f"边路由没把两个 MODEL 都给合并节点:{list(guider_inputs)}"
    merged = await ex._execute_inline_node(node_map["guider"], guider_inputs)

    m = merged["model"]
    assert m["_type"] == "flux2_model"
    assert m["spec"]["file"] == "/m/ideogram4_fp8_scaled.safe"           # 条件 DiT 作主 spec
    assert m["spec"]["adapter_arch"] == "ideogram4"
    assert m["unconditional_file"] == "/m/ideogram4_unconditional_fp8_scaled.safe"  # 第二 DiT 挂上


@pytest.mark.asyncio
async def test_dual_guider_in_executor_rejects_wrong_arch():
    """图执行里合并节点接了非 ideogram4 的 DiT → 派发前人话报错(同 #517,但走真图路径)。"""
    wf = _dual_dit_workflow()
    wf["nodes"][1]["data"]["adapter_arch"] = "flux2"  # uncond loader 误选 flux2
    ex = WorkflowExecutor(wf, runner_client=None)
    node_map = {n["id"]: n for n in wf["nodes"]}
    for nid in ("cond", "uncond"):
        ex._outputs[nid] = await ex._execute_inline_node(node_map[nid], {})
    guider_inputs = ex._get_inputs("guider")
    with pytest.raises(Exception, match="架构须为 ideogram4|架构不匹配"):
        await ex._execute_inline_node(node_map["guider"], guider_inputs)
