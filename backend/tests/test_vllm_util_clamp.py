"""clamp_util_to_free 纯函数测试 —— 不依赖 vllm 包(llm_vllm 模块可独立 import)。"""
import pytest

class TestClampUtilToFree:
    """util 按所选卡实际 free 封顶(#01:02 GPU2 启动即炸根因):
    resolve_vram_utilization 的三条路(overlay 预算/yaml/auto 公式)都以 total 为分母,
    卡上已有别的进程时 vLLM 拿到超过 free 的预算 → 启动期 OOM exit 1。"""

    def test_clamps_when_budget_exceeds_free(self):
        from src.services.inference.llm_vllm import clamp_util_to_free
        # total 24G,free 只剩 10G:0.92 预算(22G)必须被压到 (10-0.5)/24≈0.395
        out = clamp_util_to_free(0.92, gpu_free_gb=10.0, gpu_total_gb=24.0)
        assert out == pytest.approx((10.0 - 0.5) / 24.0, abs=0.01)

    def test_no_clamp_when_fits(self):
        from src.services.inference.llm_vllm import clamp_util_to_free
        out = clamp_util_to_free(0.5, gpu_free_gb=20.0, gpu_total_gb=24.0)
        assert out == 0.5

    def test_floor_when_gpu_nearly_full(self):
        from src.services.inference.llm_vllm import clamp_util_to_free
        # free 见底也不给 0/负数,留 0.05 下限让错误信息来自 vLLM 而非除零
        out = clamp_util_to_free(0.9, gpu_free_gb=0.2, gpu_total_gb=24.0)
        assert out == pytest.approx(0.05)
