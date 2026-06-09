"""split_sigmas 留噪分段切片 —— 逐数值对照 ComfyUI KSampler.sample(spec 2026-06-08 路 B PR-B2)。

用合成 sigma 表(12 步=13 值,降序 [12,11,...,1,0])验切片语义,与 ComfyUI samplers.py:1164-1175 一致。
"""
from __future__ import annotations

from src.services.inference.sigma_schedules import split_sigmas

FULL = list(range(12, -1, -1))  # [12,11,...,1,0] = 12 步的 13 个 sigma(末尾 0)


def test_no_split_returns_full():
    assert split_sigmas(FULL) == FULL


def test_base_leftover_noise_not_zeroed():
    """base #194:end_at_step=5,return_with_leftover_noise=enable(force_full_denoise=False)→
    sigmas[:6],末 sigma=7 保留(带噪交接,不置 0)。"""
    out = split_sigmas(FULL, end_at_step=5, force_full_denoise=False)
    assert out == [12, 11, 10, 9, 8, 7]
    assert out[-1] == 7  # 留噪:不是 0


def test_base_force_full_denoise_zeros_last():
    """return_with_leftover_noise=disable(force_full_denoise=True)→ 末 sigma 强制置 0。"""
    out = split_sigmas(FULL, end_at_step=5, force_full_denoise=True)
    assert out == [12, 11, 10, 9, 8, 0]


def test_refiner_start_step_slices_from_middle():
    """refiner #200:start_at_step=5 → sigmas[5:] = 从 s5=7 到 0(续采)。"""
    out = split_sigmas(FULL, start_at_step=5)
    assert out == [7, 6, 5, 4, 3, 2, 1, 0]
    assert out[-1] == 0  # 续采跑到底


def test_split_5_7_covers_full_denoise():
    """base[:6] 末=7,refiner[5:] 首=7 —— 同一 sigma 表上交接点一致(留噪接力数值连续)。"""
    base = split_sigmas(FULL, end_at_step=5, force_full_denoise=False)
    refiner = split_sigmas(FULL, start_at_step=5)
    assert base[-1] == refiner[0] == 7


def test_start_step_past_end_returns_tail():
    out = split_sigmas(FULL, start_at_step=99)
    assert out == [0]


def test_end_at_step_ge_steps_no_truncation():
    # end_at_step >= len-1 → 不截断(ComfyUI: last_step < len(sigmas)-1 才切)。
    assert split_sigmas(FULL, end_at_step=12) == FULL
    assert split_sigmas(FULL, end_at_step=99) == FULL
