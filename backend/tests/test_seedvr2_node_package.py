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


def _nodes() -> dict:
    return yaml.safe_load((_PKG / "node.yaml").read_text())["nodes"]


def _node_def(name: str = "seedvr2_upscale") -> dict:
    return _nodes()[name]


def test_seedvr2_package_three_nodes():
    """三节点对齐 ComfyUI:load_dit + load_vae(inline 产配置)+ upscale(dispatch)。"""
    cfg = yaml.safe_load((_PKG / "node.yaml").read_text())
    assert cfg["name"] == "seedvr2"
    for n in ("seedvr2_load_dit", "seedvr2_load_vae", "seedvr2_upscale"):
        assert n in cfg["nodes"], f"缺节点 {n}"


def test_seedvr2_upscale_ports():
    """增强节点:image + dit + vae 输入(dit/vae 是新端口类型)→ image 输出。"""
    nd = _node_def()
    assert [(p["id"], p["type"]) for p in nd["inputs"]] == [
        ("image", "image"), ("dit", "seedvr2_dit"), ("vae", "seedvr2_vae")]
    assert [(p["id"], p["type"]) for p in nd["outputs"]] == [("image", "image")]
    # dit_model widget 已移到 load_dit,不在 upscale。
    assert not any(w["name"] == "dit_model" for w in nd["widgets"]), "dit_model 应移到 load_dit"


def test_seedvr2_loaders_produce_typed_config_ports():
    """load_dit → dit(seedvr2_dit)、load_vae → vae(seedvr2_vae);各带配置 widget。"""
    dit = _node_def("seedvr2_load_dit")
    vae = _node_def("seedvr2_load_vae")
    # torch_compile 节点接入后,loader 各带一个可选 compile 输入(不连=不编译)。
    assert [(p["id"], p["type"]) for p in dit["inputs"]] == [("compile", "seedvr2_compile")]
    assert [(p["id"], p["type"]) for p in dit["outputs"]] == [("dit", "seedvr2_dit")]
    assert [(p["id"], p["type"]) for p in vae["inputs"]] == [("compile", "seedvr2_compile")]
    assert [(p["id"], p["type"]) for p in vae["outputs"]] == [("vae", "seedvr2_vae")]
    dit_w = {w["name"] for w in dit["widgets"]}
    assert {"dit_model", "device", "blocks_to_swap", "swap_io_components", "offload_device", "attention_mode"} <= dit_w
    vae_w = {w["name"] for w in vae["widgets"]}
    assert {"vae_model", "encode_tiled", "encode_tile_size", "decode_tiled", "decode_tile_size", "offload_device"} <= vae_w


def test_seedvr2_has_inline_executor_for_loaders():
    """现在有 executor.py:为两个 inline loader 节点注册 EXECUTORS(增强节点仍 dispatch,不在内)。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sv2_exec", _PKG / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.EXECUTORS) == {"seedvr2_load_dit", "seedvr2_load_vae", "seedvr2_torch_compile"}
    assert "seedvr2_upscale" not in mod.EXECUTORS, "增强节点是 dispatch,不该进 EXECUTORS"


def test_seedvr2_loader_executors_build_config_dicts():
    """loader executor 把 widget → 配置 dict(adapter 串进 prepare_runner 的契约)。纯 dict,CI 安全。"""
    import asyncio
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sv2_exec2", _PKG / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dit = asyncio.run(mod.exec_load_dit(
        {"dit_model": "m.safetensors", "device": "cuda:1", "blocks_to_swap": "16",
         "swap_io_components": True, "offload_device": "cpu", "attention_mode": "sdpa"}, {}))
    assert dit["dit"]["model"] == "m.safetensors"
    assert dit["dit"]["blocks_to_swap"] == 16  # 字符串 → int
    assert dit["dit"]["device"] == "cuda:1"

    vae = asyncio.run(mod.exec_load_vae(
        {"vae_model": "v.safetensors", "encode_tiled": True, "encode_tile_size": "256",
         "decode_tiled": True, "decode_tile_size": "256"}, {}))
    assert vae["vae"]["encode_tiled"] is True
    assert vae["vae"]["encode_tile_size"] == 256


def test_seedvr2_upscale_full_parity_widgets():
    """增强节点 widget 对齐 ComfyUI:含 control_after_generate(seed 控制)+ uniform_batch_size +
    temporal_overlap + prepend_frames + offload_device + enable_debug。"""
    names = {w["name"] for w in _node_def("seedvr2_upscale")["widgets"]}
    for w in ("control_after_generate", "uniform_batch_size", "temporal_overlap",
              "prepend_frames", "offload_device", "enable_debug"):
        assert w in names, f"增强节点缺 widget {w}(ComfyUI 有)"
    # control_after_generate 是 seed 控制(同 KSampler;applySeedControl 通用,纯前端)。
    cag = next(w for w in _node_def("seedvr2_upscale")["widgets"] if w["name"] == "control_after_generate")
    assert {o["value"] for o in cag["options"]} == {"fixed", "increment", "decrement", "randomize"}


def test_seedvr2_dit_widget_is_dynamic_disk_aware():
    """dit_model(在 load_dit 节点)是动态混合下拉(seedvr2_model_select),不写死 options,默认 = DEFAULT_DIT。"""
    from src.services.inference.image_seedvr2 import DEFAULT_DIT  # noqa: PLC0415

    dit_widget = next(w for w in _node_def("seedvr2_load_dit")["widgets"] if w["name"] == "dit_model")
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
