"""image-io 节点包(image_input 上传图节点)测试 —— CI 安全(读 yaml/源 + 仅测 reject 路径)。

image_input 给 image→image 节点(SeedVR2 超分)补「现有图作来源」:上传 base64 → 落盘签 URL
→ image 端口。验:① 包/节点定义 ② image 输出端口(接下游 image 输入)③ inline(无 dispatch)
④ executor 注册 + 缺图 reject(reject 在任何重 import 前抛,CI 可跑)。
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

_PKG = pathlib.Path(__file__).parent.parent / "nodes" / "image-io"


def _node_def() -> dict:
    cfg = yaml.safe_load((_PKG / "node.yaml").read_text())
    return cfg["nodes"]["image_input"]


def test_image_io_package_yaml_valid():
    cfg = yaml.safe_load((_PKG / "node.yaml").read_text())
    assert cfg["name"] == "image-io"
    assert "image_input" in cfg["nodes"]


def test_image_input_outputs_image_port():
    """无输入 + 单 image 输出(type image,接下游 SeedVR2 / image_output 的 image 输入)。"""
    nd = _node_def()
    assert nd["inputs"] == []
    assert [p["id"] for p in nd["outputs"]] == ["image"]
    assert [p["type"] for p in nd["outputs"]] == ["image"]
    assert nd["category"] == "image"
    # 有 image_upload widget(上传图存进 data.image)。
    assert any(w["widget"] == "image_upload" and w["name"] == "image" for w in nd["widgets"])


def test_image_input_is_inline_not_dispatch():
    """image_input 是 inline(CPU:解码+写盘,主进程)—— 不在 GPU dispatch 白名单。"""
    from src.services.node_routing import node_exec_class

    assert node_exec_class("image_input") == "inline"


def test_image_input_executor_registered_and_rejects_missing_image():
    """executor.py 注册 image_input;缺/非 data-URI 输入 → RuntimeError(reject 在重 import 前)。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_imgio_exec", _PKG / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "image_input" in mod.EXECUTORS

    import asyncio
    with pytest.raises(RuntimeError, match="未上传图"):
        asyncio.run(mod.exec_image_input({}, {}))
    with pytest.raises(RuntimeError, match="未上传图"):
        asyncio.run(mod.exec_image_input({"image": "/not/a/data/uri.png"}, {}))


def _ref_join_def() -> dict:
    return yaml.safe_load((_PKG / "node.yaml").read_text())["nodes"]["image_ref_join"]


def test_image_ref_join_node_two_in_one_out():
    """参考图合并:两路 image 输入 + 单 image 输出(可串联;接 KSampler 多参考编辑)。"""
    nd = _ref_join_def()
    assert [(p["id"], p["type"]) for p in nd["inputs"]] == [("image_a", "image"), ("image_b", "image")]
    assert [(p["id"], p["type"]) for p in nd["outputs"]] == [("image", "image")]
    assert nd["category"] == "image"


def test_image_ref_join_is_inline():
    """合并是纯字符串拼接(主进程),不进 GPU dispatch。"""
    from src.services.node_routing import node_exec_class

    assert node_exec_class("image_ref_join") == "inline"


def test_image_ref_join_executor_joins_passes_and_rejects():
    """两路 → 逗号串;单路透传(半成品工作流不崩);串联(上游已是逗号串)原样拼;全空报人话错误。"""
    import asyncio
    import importlib.util

    spec = importlib.util.spec_from_file_location("_imgio_exec3", _PKG / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "image_ref_join" in mod.EXECUTORS

    out = asyncio.run(mod.exec_image_ref_join({}, {"image_a": "/u/a.png?sig=1", "image_b": "/u/b.png?sig=2"}))
    assert out == {"image_url": "/u/a.png?sig=1,/u/b.png?sig=2"}
    # 单路透传(A 或 B 任一)
    assert asyncio.run(mod.exec_image_ref_join({}, {"image_a": "/u/a.png"})) == {"image_url": "/u/a.png"}
    assert asyncio.run(mod.exec_image_ref_join({}, {"image_b": "/u/b.png"})) == {"image_url": "/u/b.png"}
    # 串联:上游 join 的逗号串再拼第三张
    out = asyncio.run(mod.exec_image_ref_join({}, {"image_a": "/u/a.png,/u/b.png", "image_b": "/u/c.png"}))
    assert out == {"image_url": "/u/a.png,/u/b.png,/u/c.png"}
    with pytest.raises(RuntimeError, match="两路输入都没有图"):
        asyncio.run(mod.exec_image_ref_join({}, {}))


def _compare_def() -> dict:
    return yaml.safe_load((_PKG / "node.yaml").read_text())["nodes"]["image_compare"]


def test_image_compare_node_two_image_inputs_sink():
    """图像对比:两路 image 输入(image_a/image_b)+ 无输出(显示型 sink,前端滑动对比)。"""
    nd = _compare_def()
    assert [(p["id"], p["type"]) for p in nd["inputs"]] == [("image_a", "image"), ("image_b", "image")]
    assert nd["outputs"] == []
    assert nd["category"] == "image"


def test_image_compare_is_inline_with_noop_executor():
    """image_compare 是 inline(纯前端对比,后端 no-op);executor 注册且不抛(透传两路 url)。"""
    import asyncio
    import importlib.util

    from src.services.node_routing import node_exec_class
    assert node_exec_class("image_compare") == "inline"

    spec = importlib.util.spec_from_file_location("_imgio_exec2", _PKG / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "image_compare" in mod.EXECUTORS
    # no-op:空输入不抛(对比纯前端)
    out = asyncio.run(mod.exec_image_compare({}, {}))
    assert "image_a_url" in out and "image_b_url" in out
