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


def test_seedvr2_dit_widget_is_dynamic_disk_aware():
    """dit_model 是动态混合下拉(seedvr2_model_select)—— 选项由后端 /components/seedvr2-dit
    动态给(盘上标已就绪 / 白名单其余标可下载),node.yaml 不再写死 options(避免两份白名单漂)。
    默认 = DEFAULT_DIT。"""
    from src.services.inference.image_seedvr2 import DEFAULT_DIT  # noqa: PLC0415

    dit_widget = next(w for w in _node_def()["widgets"] if w["name"] == "dit_model")
    assert dit_widget["widget"] == "seedvr2_model_select"
    assert "options" not in dit_widget, "动态 widget 不该在 node.yaml 写死 options"
    assert dit_widget["default"] == DEFAULT_DIT, "node.yaml 默认 DiT 与 adapter DEFAULT_DIT 不一致"


def test_seedvr2_dit_whitelist_single_source_in_registry():
    """单一真相:image_seedvr2.SEEDVR2_DIT_MODELS 的每个 filename 都在 NumZ 白名单,且含 DEFAULT_DIT。
    白名单用源码文本检查(model_registry.py import 链拉 torch — CI mock torch 会炸)。"""
    from src.services.inference.image_seedvr2 import DEFAULT_DIT, SEEDVR2_DIT_MODELS  # noqa: PLC0415

    registry_src = (
        pathlib.Path(__file__).parent.parent
        / "src/services/inference/seedvr2_vendor/src/utils/model_registry.py"
    ).read_text()
    names = [m["filename"] for m in SEEDVR2_DIT_MODELS]
    assert DEFAULT_DIT in names, "DEFAULT_DIT 不在 SEEDVR2_DIT_MODELS"
    for m in SEEDVR2_DIT_MODELS:
        assert f'"{m["filename"]}":' in registry_src, f"DiT {m['filename']} 不在 NumZ 白名单"
        assert m["label"] and m["desc"], "每个 DiT 要有 label/desc 给 UI"


def test_seedvr2_dit_disk_status_marks_present_and_downloadable():
    """seedvr2_dit_models_with_disk_status:盘上有的 present=True+size_mb,缺的 present=False。
    用临时目录放一个白名单文件验证(不依赖真模型盘)。"""
    import tempfile

    from src.services.inference.image_seedvr2 import (  # noqa: PLC0415
        DEFAULT_DIT,
        seedvr2_dit_models_with_disk_status,
    )

    with tempfile.TemporaryDirectory() as td:
        (pathlib.Path(td) / DEFAULT_DIT).write_bytes(b"x" * 2048)  # 假装默认模型已下
        rows = seedvr2_dit_models_with_disk_status(model_dir=td)
        by_name = {r["filename"]: r for r in rows}
        assert by_name[DEFAULT_DIT]["present"] is True
        assert by_name[DEFAULT_DIT]["size_mb"] is not None
        assert by_name[DEFAULT_DIT]["is_default"] is True
        # 其余白名单文件不在该临时目录 → 可下载
        others = [r for r in rows if r["filename"] != DEFAULT_DIT]
        assert others and all(r["present"] is False for r in others)


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
