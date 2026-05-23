"""PR-2 T1:dequant_and_convert 共享桥接 helper(CI 安全,注入 fake diffusers 内部模块)。

实际转换正确性由 spike v3(真模型,missing=0 出正确图)+ PR-2 T5 真模型 smoke 验;
此处测「链式 dispatch→convert」+ guard,不碰真 diffusers/torch。
"""
from __future__ import annotations

import sys
import types

import pytest

from src.services.inference import quant_loaders
from src.services.inference.component_spec import ComponentSpec


def _spec() -> ComponentSpec:
    return ComponentSpec(kind="unet", file="/m/fp8mixed.safetensors", device="cpu",
                         dtype="bfloat16", adapter_arch="flux2")


def _inject_diffusers(monkeypatch, sfu: types.ModuleType) -> None:
    """注入 diffusers 父包链 → `from diffusers.loaders.single_file_utils import ...`
    解析到 fake,不 import 真 diffusers(CI 无真 torch)。"""
    for name, mod in [
        ("diffusers", types.ModuleType("diffusers")),
        ("diffusers.loaders", types.ModuleType("diffusers.loaders")),
        ("diffusers.loaders.single_file_utils", sfu),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_chains_dispatch_then_convert(monkeypatch):
    monkeypatch.setattr(quant_loaders.QUANT_LOADERS, "dispatch",
                        lambda spec: {"double_blocks.0.img_attn.qkv.weight": "DEQ"})
    captured = {}
    sfu = types.ModuleType("diffusers.loaders.single_file_utils")

    def _conv(sd):
        captured["in"] = dict(sd)
        return {"transformer.transformer_blocks.0.attn.to_q.weight": "CONV"}

    sfu.convert_flux2_transformer_checkpoint_to_diffusers = _conv
    _inject_diffusers(monkeypatch, sfu)

    out = quant_loaders.dequant_and_convert(_spec())
    assert out == {"transformer.transformer_blocks.0.attn.to_q.weight": "CONV"}
    # ① 反量化结果喂给 ② 转键
    assert captured["in"] == {"double_blocks.0.img_attn.qkv.weight": "DEQ"}


def test_guard_when_converter_missing(monkeypatch):
    monkeypatch.setattr(quant_loaders.QUANT_LOADERS, "dispatch", lambda spec: {"x": "y"})
    sfu = types.ModuleType("diffusers.loaders.single_file_utils")  # 故意不带 convert 函数
    _inject_diffusers(monkeypatch, sfu)

    with pytest.raises(ValueError, match="convert_flux2|diffusers"):
        quant_loaders.dequant_and_convert(_spec())
