"""vLLM 健康看门狗(self-heal ConnectError)单测 —— CI 可跑,全 mock,无 GPU/真 vLLM。

覆盖:① 只对 ConnectError 动手(ReadTimeout/健康不动)② 连续 fail_threshold 次才 reconcile
(防抖)③ resident→reload / 非 resident→只 unload ④ 端口存在才纳入巡检 ⑤ 模型不再 loaded 时
清计数 ⑥ notify 回调。
"""
from __future__ import annotations

import pytest

from src.services import vllm_watchdog as wd


class _Adapter:
    def __init__(self, loaded=True, port=40000):
        self.is_loaded = loaded
        self._port = port


class _Spec:
    def __init__(self, resident=False):
        self.resident = resident


class _Entry:
    def __init__(self, loaded=True, port=40000, resident=False):
        self.adapter = _Adapter(loaded, port)
        self.spec = _Spec(resident)


class _MM:
    """最小 model_manager 替身:_models + unload/load 记账。"""
    def __init__(self, models):
        self._models = models
        self.unloaded: list[tuple[str, bool]] = []
        self.loaded: list[str] = []

    async def unload_model(self, mid, force=False):
        self.unloaded.append((mid, force))
        self._models.pop(mid, None)  # 真 unload 会从 _models 摘掉

    async def load_model(self, mid):
        self.loaded.append(mid)


def _snap(name, healthy, error=None):
    return {"name": name, "port": 1, "healthy": healthy, "config": {}, "stats": {}, "error": error}


# ---------- llm_targets ----------

def test_targets_only_loaded_with_port():
    mm = _MM({
        "a": _Entry(loaded=True, port=40001),
        "b": _Entry(loaded=False, port=40002),   # 未加载 → 排除
        "c": _Entry(loaded=True, port=None),      # 无端口(非 vLLM)→ 排除
    })
    assert wd.llm_targets(mm) == [("a", 40001)]


# ---------- run_cycle 判据 ----------

@pytest.mark.asyncio
async def test_healthy_no_action():
    mm = _MM({"a": _Entry()})
    fails: dict[str, int] = {}
    async def snap(t): return [_snap("a", True)]
    out = await wd.run_cycle(mm, fails, snapshot_fn=snap)
    assert out == [] and fails == {} and mm.unloaded == []


@pytest.mark.asyncio
async def test_readtimeout_not_treated_as_dead():
    """vLLM 忙(ReadTimeout)不是死,绝不能 reconcile。"""
    mm = _MM({"a": _Entry()})
    fails: dict[str, int] = {}
    async def snap(t): return [_snap("a", False, "ReadTimeout")]
    out = await wd.run_cycle(mm, fails, snapshot_fn=snap)
    assert out == [] and fails == {} and mm.unloaded == []


@pytest.mark.asyncio
async def test_connecterror_debounced_then_reconcile():
    """ConnectError 第一次只计数,达到阈值才动手(防 startup/restart 瞬时窗口)。"""
    mm = _MM({"a": _Entry(resident=True)})
    fails: dict[str, int] = {}
    async def snap(t): return [_snap("a", False, "ConnectError")]

    out1 = await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=snap)
    assert out1 == [] and fails == {"a": 1} and mm.unloaded == []   # 第一次只计数

    out2 = await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=snap)
    assert out2 == [("a", "reloaded")]                              # 第二次 reconcile
    assert mm.unloaded == [("a", True)] and mm.loaded == ["a"]      # force-unload + reload
    assert fails == {}                                             # 动手后清零


@pytest.mark.asyncio
async def test_resident_reloads_nonresident_only_unloads():
    # resident → reload
    mm_r = _MM({"a": _Entry(resident=True)})
    assert await wd.reconcile_dead(mm_r, "a") == "reloaded"
    assert mm_r.unloaded == [("a", True)] and mm_r.loaded == ["a"]
    # 非 resident → 只清陈旧态,不复活
    mm_n = _MM({"a": _Entry(resident=False)})
    assert await wd.reconcile_dead(mm_n, "a") == "unloaded"
    assert mm_n.unloaded == [("a", True)] and mm_n.loaded == []


@pytest.mark.asyncio
async def test_recovery_resets_counter():
    """死一次(计数1)后又恢复 → 计数清零,不会累计到阈值误杀。"""
    mm = _MM({"a": _Entry(resident=True)})
    fails: dict[str, int] = {}
    async def dead(t): return [_snap("a", False, "ConnectError")]
    async def alive(t): return [_snap("a", True)]
    await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=dead)
    assert fails == {"a": 1}
    await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=alive)
    assert fails == {} and mm.unloaded == []


@pytest.mark.asyncio
async def test_counter_dropped_when_model_gone():
    """模型不再 loaded(被正常卸载)→ 残留计数被清,不泄漏。"""
    mm = _MM({"a": _Entry()})
    fails = {"a": 1, "stale": 5}
    async def snap(t): return [_snap("a", True)]
    await wd.run_cycle(mm, fails, snapshot_fn=snap)
    assert "stale" not in fails


@pytest.mark.asyncio
async def test_notify_called_on_reconcile():
    mm = _MM({"a": _Entry(resident=True)})
    fails = {"a": 1}
    seen: list[tuple[str, str]] = []
    async def snap(t): return [_snap("a", False, "ConnectError")]
    async def notify(mid, action): seen.append((mid, action))
    await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=snap, notify=notify)
    assert seen == [("a", "reloaded")]


@pytest.mark.asyncio
async def test_reconcile_failure_does_not_crash_cycle():
    """单个 reconcile 抛错不能让整轮挂掉,且计数清零(下轮重试)。"""
    mm = _MM({"a": _Entry(resident=True)})
    fails = {"a": 1}
    async def snap(t): return [_snap("a", False, "ConnectError")]
    async def boom(mgr, mid): raise RuntimeError("load failed")
    out = await wd.run_cycle(mm, fails, fail_threshold=2, snapshot_fn=snap, reconcile_fn=boom)
    assert out == [] and fails == {}   # 没抛出,计数清零
