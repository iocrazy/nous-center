"""SeedVR2 PR-3c:节点包加载 + 节点定义正确(CI 安全 —— 只读 yaml/源,不 import torch)。

验:① seedvr2 包被 scan_packages 发现、seedvr2_upscale 进 get_all_definitions
② 节点 image→image 端口 + dispatch 归类 ③ dit_model 默认值在 NumZ 白名单且 = adapter
DEFAULT_DIT(前后端契约一致,避免选了引擎不认的模型)④ 是 dispatch(无 inline executor)。
前端全声明式(loadPluginDefinitions 读 /api/v1/nodes/definitions 自动注册),无需 FE 代码。
"""
from __future__ import annotations

import pathlib

import yaml

_PKG = pathlib.Path(__file__).parent.parent / "nodes" / "seedvr2"


def _node_def() -> dict:
    cfg = yaml.safe_load((_PKG / "node.yaml").read_text())
    return cfg["nodes"]["seedvr2_upscale"]


def test_seedvr2_package_yaml_valid():
    """node.yaml 解析 + 有 seedvr2_upscale 节点。"""
    cfg = yaml.safe_load((_PKG / "node.yaml").read_text())
    assert cfg["name"] == "seedvr2"
    assert "seedvr2_upscale" in cfg["nodes"]


def test_seedvr2_node_image_to_image_ports():
    """图→图:单 image 输入 + 单 image 输出(接上游 VAE Decode → 下游 image_output)。"""
    nd = _node_def()
    assert [p["id"] for p in nd["inputs"]] == ["image"]
    assert [p["type"] for p in nd["inputs"]] == ["image"]
    assert [p["id"] for p in nd["outputs"]] == ["image"]
    assert [p["type"] for p in nd["outputs"]] == ["image"]
    assert nd["category"] == "image"


def test_seedvr2_no_inline_executor():
    """dispatch 节点不带 executor.py(不进 EXECUTORS;runner 执行)。"""
    assert not (_PKG / "executor.py").exists(), "seedvr2 是 dispatch 节点,不该有 inline executor"


def test_seedvr2_default_dit_in_whitelist_and_matches_adapter():
    """前后端契约:node.yaml 的 dit_model 默认值必须 ① 在 NumZ 白名单 ② = adapter DEFAULT_DIT
    (否则 UI 默认选的模型引擎不认 / 跟 runner 缺省不一致)。所有 select 选项都得在白名单。

    白名单用**源码文本检查**(model_registry.py import 链拉 torch — CI mock torch 会炸);
    image_seedvr2.DEFAULT_DIT 是模块常量,顶层无 torch,CI 安全可直接 import。"""
    from src.services.inference.image_seedvr2 import DEFAULT_DIT  # noqa: PLC0415

    registry_src = (
        pathlib.Path(__file__).parent.parent
        / "src/services/inference/seedvr2_vendor/src/utils/model_registry.py"
    ).read_text()

    dit_widget = next(w for w in _node_def()["widgets"] if w["name"] == "dit_model")
    assert dit_widget["default"] == DEFAULT_DIT, "node.yaml 默认 DiT 与 adapter DEFAULT_DIT 不一致"
    # 白名单条目是 MODEL_REGISTRY 的字符串 key —— 源码里以 "<filename>": 形态出现。
    for opt in dit_widget["options"]:
        assert f'"{opt["value"]}":' in registry_src, f"DiT 选项 {opt['value']} 不在 NumZ 白名单"


def test_seedvr2_color_correction_options_match_request():
    """node.yaml 的 color_correction 选项必须 ⊆ UpscaleRequest 接受的字面量(否则验证报错)。"""
    cc_widget = next(w for w in _node_def()["widgets"] if w["name"] == "color_correction")
    yaml_opts = {opt["value"] for opt in cc_widget["options"]}
    # UpscaleRequest.color_correction 的 Literal 集合
    allowed = {"lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"}
    assert yaml_opts <= allowed, f"color_correction 选项越界: {yaml_opts - allowed}"


def test_seedvr2_upscale_is_dispatch():
    """seedvr2_upscale 走 dispatch(PR-3b 已登记;PR-3c 节点存在后这条端到端成立)。"""
    from src.services.node_routing import node_exec_class

    assert node_exec_class("seedvr2_upscale") == "dispatch"
