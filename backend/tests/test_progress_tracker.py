"""ProgressTracker 单测(PR-4)— 锁定:stage/step/tick/finish + throttle + 滑窗 + 兼容降级。"""
from __future__ import annotations

import time

import pytest

from src.services.inference.progress_tracker import ProgressTracker, emit_to_callback


def _collect():
    events: list[dict] = []

    def cb(done: int, total: int, **extras):
        events.append({"done": done, "total": total, **extras})
    return events, cb


def test_no_callback_is_noop():
    pt = ProgressTracker(None)
    pt.stage("a")
    pt.step(1, 10)
    pt.tick(1.0, 10.0)
    pt.finish(10)
    # 没崩就行 — callback None 时所有 API no-op


def test_stage_emits_with_zero_progress_default():
    events, cb = _collect()
    pt = ProgressTracker(cb)
    pt.stage("text_encode")
    assert len(events) == 1
    ev = events[0]
    assert ev["stage"] == "text_encode"
    assert ev["progress"] == 0.0
    assert ev["detail"] == "text_encode"
    assert ev["eta_ms"] is None  # stage 切换不算 ETA
    assert ev["step_latency_ms"] is None


def test_step_emits_with_latency_and_eta():
    events, cb = _collect()
    pt = ProgressTracker(cb, stage="dit_denoise")
    # 3 步,每步 sleep ~10ms
    for i in range(1, 4):
        time.sleep(0.01)
        pt.step(i, 10)
    assert len(events) == 3
    last = events[-1]
    assert last["stage"] == "dit_denoise"
    assert last["done"] == 3 and last["total"] == 10
    assert last["progress"] == 3 / 10  # 0.3 < 0.95
    # latency 滑窗均值 ≈ 10ms(允许 jitter)
    assert 5 <= last["step_latency_ms"] <= 100
    # ETA = (10-3) × avg
    assert last["eta_ms"] == last["step_latency_ms"] * 7


def test_step_progress_capped_at_0_95_before_finish():
    events, cb = _collect()
    pt = ProgressTracker(cb)
    pt.step(10, 10)  # 100% 应被 cap 到 0.95
    assert events[-1]["progress"] == 0.95


def test_finish_emits_progress_1_and_eta_0():
    events, cb = _collect()
    pt = ProgressTracker(cb, stage="tts_synth")
    pt.step(5, 10)
    pt.finish(total_units=10)
    last = events[-1]
    assert last["progress"] == 1.0
    assert last["eta_ms"] == 0
    assert last["done"] == 10 and last["total"] == 10
    assert "done" in last["detail"]


def test_finish_idempotent():
    events, cb = _collect()
    pt = ProgressTracker(cb)
    pt.finish(10)
    n = len(events)
    pt.finish(10)
    pt.step(5, 10)
    assert len(events) == n  # finished 后 no-op


def test_throttle_skips_intermediate_steps():
    """throttle_ms=100,5 步内全部 < 100ms 应只发 1 次(首步)+ 末帧 finish。"""
    events, cb = _collect()
    pt = ProgressTracker(cb, throttle_ms=100)
    for i in range(1, 6):
        pt.step(i, 10)  # 极快连续,delta_ms ≈ 0
    pt.finish(10)
    # 首步不受 throttle(last_emit_t=0,任何 now-0 > 100ms);其余被 cap;末帧 finish 不受 throttle
    intermediate = [e for e in events if e["progress"] < 1.0]
    assert len(intermediate) == 1
    assert events[-1]["progress"] == 1.0


def test_throttle_allows_after_window():
    """throttle 窗口外的 step 可以发。"""
    events, cb = _collect()
    pt = ProgressTracker(cb, throttle_ms=50)
    pt.step(1, 10)
    time.sleep(0.08)  # 80ms > 50ms
    pt.step(2, 10)
    # 至少 2 帧
    assert len([e for e in events if e["progress"] < 1.0]) >= 2


def test_tick_estimates_progress_from_elapsed():
    events, cb = _collect()
    pt = ProgressTracker(cb, stage="tts_synth")
    pt.tick(elapsed_s=2.0, estimated_total_s=10.0)
    ev = events[-1]
    assert ev["stage"] == "tts_synth"
    assert ev["progress"] == 0.2
    assert ev["eta_ms"] == 8000
    assert ev["done"] == 2 and ev["total"] == 10


def test_tick_progress_capped_at_0_95():
    events, cb = _collect()
    pt = ProgressTracker(cb)
    pt.tick(elapsed_s=11.0, estimated_total_s=10.0)  # 110% est
    assert events[-1]["progress"] == 0.95


def test_preview_url_forwarded():
    events, cb = _collect()
    pt = ProgressTracker(cb)
    pt.step(1, 10, preview_url="data:image/jpeg;base64,xxx")
    assert events[-1]["preview_url"] == "data:image/jpeg;base64,xxx"


def test_legacy_callback_signature_done_total_only():
    """老 fake 只接 (done, total) — TypeError 降级到最简形态。"""
    calls: list[tuple[int, int]] = []

    def legacy_cb(done: int, total: int) -> None:
        calls.append((done, total))

    pt = ProgressTracker(legacy_cb)
    pt.stage("text_encode")
    pt.step(3, 10)
    pt.finish(10)
    # 全部都通过降级落到 (done, total)
    assert len(calls) == 3
    assert calls[-1] == (10, 10)


def test_legacy_callback_signature_with_preview_url():
    """老 PR-F 契约只接 (done, total, preview_url=) — TypeError 降级 1。"""
    calls: list[dict] = []

    def cb(done: int, total: int, preview_url=None):
        calls.append({"done": done, "total": total, "preview_url": preview_url})

    pt = ProgressTracker(cb)
    pt.step(1, 10, preview_url="data:image/jpeg;base64,foo")
    assert calls[-1]["preview_url"] == "data:image/jpeg;base64,foo"


def test_stage_override_in_step():
    """step() 可临时覆盖 default stage。"""
    events, cb = _collect()
    pt = ProgressTracker(cb, stage="dit_denoise")
    pt.step(1, 10, stage="other_stage")
    assert events[-1]["stage"] == "other_stage"


def test_latency_window_smooths_spike():
    """首步 spike 在滑窗后被平滑。"""
    events, cb = _collect()
    pt = ProgressTracker(cb, latency_window=4)
    # 首步 100ms,后续 10ms
    time.sleep(0.10)
    pt.step(1, 5)
    spike_eta = events[-1]["eta_ms"]
    for i in range(2, 5):
        time.sleep(0.01)
        pt.step(i, 5)
    # 后期 eta(基于滑窗均值)应低于首步
    later_eta = events[-1]["eta_ms"]
    assert later_eta < spike_eta


# —— emit_to_callback:signature.bind 判兼容,不吞回调 body 的真 TypeError(审查 footgun)——
def test_emit_body_typeerror_not_swallowed():
    """回调签名匹配但 body 真抛 TypeError → 必须冒泡,不被当降级信号吞掉/重调。"""
    calls = []

    def cb(done, total, **extras):
        calls.append((done, total))
        raise TypeError("real bug inside callback body")

    with pytest.raises(TypeError, match="real bug inside callback body"):
        emit_to_callback(cb, 1, 2, [{"stage": "x"}])
    assert len(calls) == 1, "只该调一次(没因吞 TypeError 降级重调)"


def test_emit_signature_mismatch_falls_back_to_min():
    """回调只接 (done, total) → 降级到最简契约,不抛。"""
    calls = []

    def cb(done, total):
        calls.append((done, total))

    emit_to_callback(cb, 3, 4, [{"stage": "x", "progress": 0.5}])
    assert calls == [(3, 4)]


def test_emit_full_signature_gets_extras():
    """回调接 **extras → 收到富参数(第一候选)。"""
    got = {}

    def cb(done, total, **extras):
        got.update(extras)
        got["dt"] = (done, total)

    emit_to_callback(cb, 5, 6, [{"stage": "denoise", "progress": 0.9}])
    assert got["dt"] == (5, 6)
    assert got["stage"] == "denoise" and got["progress"] == 0.9


def test_emit_middle_candidate_preview_url():
    """只接 preview_url(老 PR-F 契约)→ 命中第二候选。"""
    got = {}

    def cb(done, total, preview_url=None):
        got["preview_url"] = preview_url

    emit_to_callback(cb, 1, 1, [{"stage": "x"}, {"preview_url": "http://p"}])
    assert got["preview_url"] == "http://p"
