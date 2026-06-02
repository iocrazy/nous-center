"""Lane S: 节点分流判定（dispatch vs inline-HTTP）。"""
import pytest

from src.services.node_routing import node_exec_class, DISPATCH_NODE_TYPES


def test_llm_node_is_inline():
    """llm 节点走 inline —— 现有 LLMNode 已经 HTTP 调 vLLM。"""
    assert node_exec_class("llm") == "inline"


def test_tts_engine_is_dispatch():
    assert node_exec_class("tts_engine") == "dispatch"


@pytest.mark.parametrize("node_type", [
    "text_input", "text_output", "prompt_template", "agent",
    "if_else", "python_exec", "passthrough", "output",
])
def test_cpu_nodes_are_inline(node_type):
    """纯 CPU / 逻辑节点走 inline。"""
    assert node_exec_class(node_type) == "inline"


def test_unknown_node_defaults_inline():
    """未知节点类型（含插件节点）默认 inline —— 保守：不假设它需要 GPU runner。"""
    assert node_exec_class("some_plugin_node") == "inline"


def test_dispatch_set_is_explicit():
    """DISPATCH_NODE_TYPES 是显式白名单，新增 GPU 节点必须在此登记。"""
    assert DISPATCH_NODE_TYPES == {"tts_engine", "flux2_vae_decode", "seedvr2_upscale"}


def test_seedvr2_upscale_is_dispatch():
    """SeedVR2 超分吃 GPU,走 dispatch(image runner 组,SeedVR2 PR-3b)。"""
    assert node_exec_class("seedvr2_upscale") == "dispatch"


def test_flux2_vae_decode_is_dispatch():
    """收敛后 flux2_vae_decode 走 dispatch(整模型在 runner 子进程执行)。"""
    from src.services.node_routing import node_exec_class
    assert node_exec_class("flux2_vae_decode") == "dispatch"


def test_flux2_inline_nodes():
    from src.services.node_routing import node_exec_class
    for t in ("flux2_load_diffusion_model", "flux2_load_clip", "flux2_load_vae",
              "flux2_load_lora", "flux2_encode_prompt", "flux2_ksampler"):
        assert node_exec_class(t) == "inline"
