"""Lane G: _make_step_callback pure unit tests (no torch dependency).

These are split out from test_image_adapter_cancel.py so they can run in
environments without the `image` extra (torch) installed. The callback
itself doesn't touch torch — only the surrounding sample()/infer() do.
"""
import pytest

from src.services.inference.cancel_flag import CancelFlag
from src.services.inference.exceptions import NodeCancelled
from src.services.inference.image_diffusers import _make_step_callback


def test_step_callback_passthrough_when_not_cancelled():
    """flag 未 set 时，callback 原样返回 callback_kwargs，不抛。"""
    flag = CancelFlag()
    cb = _make_step_callback(flag)
    kwargs = {"latents": "fake-tensor"}
    out = cb(None, 3, 100, kwargs)
    assert out is kwargs  # 原样透传


def test_step_callback_raises_when_cancelled():
    """flag 已 set 时，callback 抛 NodeCancelled，reason 来自 flag。"""
    flag = CancelFlag()
    flag.set("user requested")
    cb = _make_step_callback(flag)
    with pytest.raises(NodeCancelled) as ei:
        cb(None, 5, 80, {})
    assert ei.value.reason == "user requested"


def test_step_callback_none_flag_never_raises():
    """cancel_flag=None（V1 兼容路径）时 callback 永不抛。"""
    cb = _make_step_callback(None)
    out = cb(None, 1, 10, {"x": 1})
    assert out == {"x": 1}
