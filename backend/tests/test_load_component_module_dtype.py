"""PR-1 T2: weight_dtype=default → from_pretrained 不传 torch_dtype(文件原生精度)。"""
from __future__ import annotations

from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import _torch_dtype_from


def test_torch_dtype_default_is_none():
    # default ⇒ None ⇒ 调用方 from_pretrained 省略 torch_dtype(原生)
    assert _torch_dtype_from("default") is None


def test_torch_dtype_known():
    import torch
    assert _torch_dtype_from("bfloat16") == torch.bfloat16
    assert _torch_dtype_from("float16") == torch.float16
    assert _torch_dtype_from("fp8_e4m3") == torch.float8_e4m3fn


def test_componentspec_accepts_default_dtype():
    s = ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:0", dtype="default")
    assert s.dtype == "default"
