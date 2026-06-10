"""SeedVR2 device 串归一(_resolve_dev)—— 修「device string: auto」RuntimeError。

节点 widget device 默认 "auto",model_manager 解析出 cuda:N 传给 self.device 却没写回
config,_load_sync 读到 "auto" 直接喂 torch.device("auto") → 抛
「device type at start of device string: auto」。_resolve_dev 兜底归一。
"""
from __future__ import annotations

from src.services.inference.image_seedvr2 import _resolve_dev


def test_auto_resolves_to_fallback():
    assert _resolve_dev("auto", "cuda:1") == "cuda:1"


def test_bare_cuda_resolves_to_fallback():
    # 裸 "cuda"(无索引)也归一到已解析的具体卡,免歧义。
    assert _resolve_dev("cuda", "cuda:2") == "cuda:2"


def test_empty_and_none_value_resolve_to_fallback():
    assert _resolve_dev("", "cuda:0") == "cuda:0"
    assert _resolve_dev(None, "cuda:0") == "cuda:0"


def test_none_string_preserved_for_offload_off():
    # "none" = offload 关,必须原样保留(下游 != "none" 判定靠它)。
    assert _resolve_dev("none", "cuda:0") == "none"


def test_concrete_device_passthrough():
    assert _resolve_dev("cuda:2", "cuda:0") == "cuda:2"
    assert _resolve_dev("cpu", "cuda:0") == "cpu"


def test_case_insensitive_auto():
    assert _resolve_dev("AUTO", "cuda:1") == "cuda:1"
