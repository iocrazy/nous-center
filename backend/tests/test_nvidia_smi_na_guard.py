"""round5:nvidia-smi `[N/A]` 字段不该让整个 GPU 列表清空 / poll 崩。"""

from unittest.mock import MagicMock

import pytest


def _fake_run(stdout):
    return MagicMock(returncode=0, stdout=stdout)


def test_gpu_stats_na_util_temp_does_not_empty_list(monkeypatch):
    """某卡 util/temp 报 [N/A] → 该卡仍在表里(降级 0),不清空所有卡。"""
    from src.api.routes import monitor
    # 两卡:卡0 正常,卡1 util+temp 为 [N/A](被动散热/特定状态)
    out = (
        "0, NVIDIA RTX 6000, 1000, 96000, 30, 45, 40, 100.0, 300.0\n"
        "1, NVIDIA RTX 3090, 2000, 24000, [N/A], [N/A], [N/A], [N/A], [N/A]\n"
    )
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: _fake_run(out))
    gpus = monitor._gpu_stats_nvidia_smi()
    assert gpus is not None
    assert len(gpus) == 2  # 老 bug:卡1 的 int([N/A]) 崩 → 整批 None/[]
    assert gpus[1]["utilization_gpu"] == 0
    assert gpus[1]["temperature"] == 0
    assert gpus[1]["memory_total_mb"] == 24000


def test_smi_int_float_guards():
    from src.api.routes.monitor import _smi_float, _smi_int
    assert _smi_int("[N/A]") == 0
    assert _smi_int("45") == 45
    assert _smi_int("") == 0
    assert _smi_int("bogus", default=-1) == -1
    assert _smi_float("[N/A]") == 0.0
    assert _smi_float("12.5") == 12.5


@pytest.mark.asyncio
async def test_poll_gpu_stats_na_does_not_crash(monkeypatch):
    import src.services.gpu_monitor as gm
    out = "0, 1000, 96000, 95000, [N/A], [N/A]\n"
    monkeypatch.setattr(gm.subprocess, "run", lambda *a, **k: _fake_run(out))
    monkeypatch.setattr(gm, "_last_poll", 0.0)
    monkeypatch.setattr(gm, "_gpu_stats", [])
    stats = gm.poll_gpu_stats()
    assert len(stats) == 1
    assert stats[0]["utilization_pct"] == 0
    assert stats[0]["free_mb"] == 95000
