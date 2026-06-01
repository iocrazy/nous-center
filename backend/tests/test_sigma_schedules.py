"""sigma_schedules port 正确性 —— 数值对齐 ComfyUI ground truth。

fixture(tests/fixtures/comfy_sigmas_shift3_steps20.json)是用 ComfyUI 自己的库
(comfy/samplers.py SCHEDULER_HANDLERS + ModelSamplingDiscreteFlow shift=3.0,steps=20)
dump 的真值。CI mock torch 但本模块纯 numpy/math 不依赖 torch,可真跑。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.inference.sigma_schedules import (
    INJECTED_SCHEDULERS,
    NATIVE_SCHEDULERS,
    SIGMA_SCHEDULES,
    compute_sigmas,
)

_GT = json.loads(
    (Path(__file__).parent / "fixtures" / "comfy_sigmas_shift3_steps20.json").read_text()
)


# 生产真正手动算 sigma 注入的是 INJECTED 这 5 个(纯 numpy/math,CI mock torch 下可跑)。
# native 4 个(normal/karras/exponential/beta)生产走 diffusers use_*_sigmas,不经本模块;
# 其中 beta 的本地参考实现依赖 scipy,scipy 在 conftest mock torch 下 import 会炸 → 不在 CI
# 跑 beta 的数值对比(它的正确性由 diffusers 保证,且本地 dev 已对齐 ground truth)。
_CI_TESTABLE = sorted(INJECTED_SCHEDULERS | {"normal", "karras", "exponential"})


@pytest.mark.parametrize("name", _CI_TESTABLE)
def test_sigmas_match_comfyui_ground_truth(name):
    """每个 scheduler 的 sigma 序列对齐 ComfyUI(max diff < 1e-4)。"""
    gt = _GT[name]
    assert isinstance(gt, list), f"{name} ComfyUI ground truth 不是 list: {gt!r}"
    ours = compute_sigmas(name, 20, shift=3.0)
    assert len(ours) == len(gt), f"{name}: 长度 {len(ours)} != GT {len(gt)}"
    max_diff = max(abs(a - b) for a, b in zip(ours, gt))
    assert max_diff < 1e-4, f"{name}: max_diff={max_diff:.2e} 偏离 ComfyUI"


def test_all_9_comfyui_schedulers_present():
    """对齐 ComfyUI 9 个 scheduler 全覆盖。"""
    assert set(SIGMA_SCHEDULES) == {
        "normal", "karras", "exponential", "beta",
        "simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal",
    }
    assert NATIVE_SCHEDULERS | INJECTED_SCHEDULERS == set(SIGMA_SCHEDULES)
    assert NATIVE_SCHEDULERS & INJECTED_SCHEDULERS == set()


def test_sigmas_end_with_zero_descending():
    """sigma 序列含末尾 0(对齐 ComfyUI)。多数单调下降(linear_quadratic 等特殊形除外)。
    跳过 beta(scipy 依赖在 conftest mock torch 下炸)。"""
    for name in SIGMA_SCHEDULES:
        if name == "beta":
            continue
        sigs = compute_sigmas(name, 20, shift=3.0)
        assert sigs[-1] == pytest.approx(0.0, abs=1e-5), f"{name} 末尾非 0"
        assert sigs[0] > 0.5, f"{name} 起点 sigma 应接近 sigma_max"


def test_unknown_scheduler_raises():
    with pytest.raises(ValueError, match="未知 scheduler"):
        compute_sigmas("nonexistent", 20)
