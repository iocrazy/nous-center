"""LCS 节点 executor — 采样期干预描述符(inline,产配置,不真标定)。

lcs_sharpness_intervene 是 **inline 节点**(主进程,无 GPU):把 widget 值 bundle 成干预描述符
`{"_type":"lcs_sharpness", strength, start_step, end_step}`,append 到上游链(intervene_in)→ 输出
list。真标定(光栅 PCA)+ per-step hook 在引擎惰性做(image_modular._build_interventions →
lcs_integration.build_sharpness_fn,用 pipe.vae + VAE 指纹缓存)。spec 2026-06-10-sampling-intervention-hook。
"""
from __future__ import annotations

from typing import Any


def _num(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


async def exec_sharpness_intervene(data: dict, inputs: dict) -> dict:
    """widget → lcs_sharpness 描述符,append 到上游干预链 → intervene list。"""
    # 引擎边界归一(2026-06-11 体检):widget 范围只是 UI 滑杆,旧 JSON/API 直发可携非法值
    # (strength=100 会把 latent 推爆)。clamp 到 widget 同款范围;合法值原样(零回归)。
    desc = {
        "_type": "lcs_sharpness",
        "strength": max(-5.0, min(5.0, _num(data.get("strength"), 1.0))),
        "start_step": max(0, int(_num(data.get("start_step"), 5))),
        "end_step": max(0, int(_num(data.get("end_step"), 15))),
    }
    return {"intervene": [*_prev_chain(inputs), desc]}


_ANCHOR_MODES = {"self_anchor", "smooth"}  # 与 node.yaml options / lcs_integration 真分支同步


async def exec_color_anchor(data: dict, inputs: dict) -> dict:
    """widget → lcs_color_anchor 描述符,append 到上游干预链 → intervene list。"""
    _mode = str(data.get("mode") or "self_anchor")
    desc = {
        "_type": "lcs_color_anchor",
        "mode": _mode if _mode in _ANCHOR_MODES else "self_anchor",
        "intensity": max(0.0, min(1.0, _num(data.get("intensity"), 0.8))),
    }
    return {"intervene": [*_prev_chain(inputs), desc]}


def _prev_chain(inputs: dict) -> list:
    prev = inputs.get("intervene_in") or inputs.get("intervene") or []
    if isinstance(prev, dict):
        return [prev]
    return prev if isinstance(prev, list) else []


EXECUTORS = {
    "lcs_sharpness_intervene": exec_sharpness_intervene,
    "lcs_color_anchor": exec_color_anchor,
}
