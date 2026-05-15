"""Lane H: F2 GPU-free 探针测试 —— nvidia-smi-backed，无 GPU 保守返回 True。"""
from src.runner.gpu_free_probe import make_gpu_free_probe


def test_probe_true_when_all_gpus_have_enough_free(monkeypatch):
    """目标 GPU 的 free_mb 全部 >= baseline → 探针返回 True。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
            {"index": 1, "free_mb": 23000, "total_mb": 24000, "used_mb": 1000,
             "utilization_pct": 2, "temperature": 38},
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is True


def test_probe_false_when_a_gpu_still_occupied(monkeypatch):
    """某个目标 GPU 的 free_mb < baseline（CUDA context 还没回收）→ False。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
            {"index": 1, "free_mb": 8000, "total_mb": 24000, "used_mb": 16000,
             "utilization_pct": 90, "temperature": 70},  # 还占着
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is False


def test_probe_true_when_no_gpu_stats(monkeypatch):
    """nvidia-smi 返回空（CUDA_VISIBLE_DEVICES='' 测试环境）→ 保守返回 True，不阻塞重启。"""
    monkeypatch.setattr("src.runner.gpu_free_probe.poll_gpu_stats", lambda: [])
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 1]) is True


def test_probe_true_when_target_gpu_missing_from_stats(monkeypatch):
    """目标 GPU index 不在 stats 里（拔卡 / 索引错位）→ 该 GPU 视为不可判定，
    保守返回 True 不卡死重启循环（宁可早重启也不无限等一个不存在的 GPU）。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 22000, "total_mb": 24000, "used_mb": 2000,
             "utilization_pct": 5, "temperature": 40},
        ],
    )
    probe = make_gpu_free_probe(baseline_free_mb=20000)
    assert probe([0, 5]) is True  # GPU 5 不存在 → 不阻塞


def test_probe_default_baseline_from_total(monkeypatch):
    """不传 baseline_free_mb → 用每卡 total_mb 的 80% 作为基线。"""
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 20000, "total_mb": 24000, "used_mb": 4000,
             "utilization_pct": 10, "temperature": 45},
        ],
    )
    # 24000 * 0.8 = 19200；free 20000 >= 19200 → True
    probe = make_gpu_free_probe()
    assert probe([0]) is True
    # 把 free 压到 19000 < 19200 → False
    monkeypatch.setattr(
        "src.runner.gpu_free_probe.poll_gpu_stats",
        lambda: [
            {"index": 0, "free_mb": 19000, "total_mb": 24000, "used_mb": 5000,
             "utilization_pct": 20, "temperature": 50},
        ],
    )
    probe2 = make_gpu_free_probe()
    assert probe2([0]) is False
