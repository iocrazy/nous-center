"""vllm_metrics.fetch_instance 的 /health 兜底(spec 2026-07-20 §7 Arc 2)。

sgl-omni(MOSS ASR)无 vLLM Prometheus /metrics 路由 → /metrics 返 404。看门狗/监控探测
必须退回 /health 判活(否则被误判 unhealthy;若再被当死会误杀 resident)。此文件锁:
① 404→/health 200 = healthy ② 404→/health 非200 = unhealthy ③ 200 metrics 正常解析(不兜底)
④ 端口连不上 = ConnectError(watchdog 唯一自愈触发判据,保持不变)。
"""
from __future__ import annotations

import httpx
import pytest

import src.services.vllm_metrics as vm


class _Resp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """按 URL 路由返回 /metrics 与 /health 的预置响应,或抛连接错误。"""

    def __init__(self, *, metrics=None, health=None, connect_error=False):
        self._metrics = metrics
        self._health = health
        self._connect_error = connect_error
        self.calls: list[str] = []

    async def get(self, url, timeout=None):
        self.calls.append(url)
        if self._connect_error:
            raise httpx.ConnectError("refused")
        if url.endswith("/metrics"):
            return self._metrics
        if url.endswith("/health"):
            return self._health
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture(autouse=True)
def _restore_client():
    saved = vm._client
    yield
    vm._client = saved


@pytest.mark.asyncio
async def test_metrics_404_falls_back_to_health_ok(monkeypatch):
    fc = _FakeClient(metrics=_Resp(404), health=_Resp(200))
    monkeypatch.setattr(vm, "_get_client", lambda: fc)
    snap = await vm.fetch_instance("moss_transcribe_diarize", 8003)
    assert snap.healthy is True
    assert snap.config.get("backend") == "sglang_omni"
    assert any(u.endswith("/health") for u in fc.calls)  # 确实退回探了 /health


@pytest.mark.asyncio
async def test_metrics_404_health_down_is_unhealthy(monkeypatch):
    fc = _FakeClient(metrics=_Resp(404), health=_Resp(500))
    monkeypatch.setattr(vm, "_get_client", lambda: fc)
    snap = await vm.fetch_instance("moss_transcribe_diarize", 8003)
    assert snap.healthy is False


@pytest.mark.asyncio
async def test_vllm_metrics_200_no_fallback(monkeypatch):
    body = "vllm:num_requests_running{model_name=\"x\"} 2.0\n"
    fc = _FakeClient(metrics=_Resp(200, body))
    monkeypatch.setattr(vm, "_get_client", lambda: fc)
    snap = await vm.fetch_instance("qwen", 40000)
    assert snap.healthy is True
    assert snap.stats.get("running") == 2.0
    assert fc.calls == ["http://127.0.0.1:40000/metrics"]  # 未探 /health


@pytest.mark.asyncio
async def test_connect_error_stays_connecterror(monkeypatch):
    # 端口连不上 → ConnectError(watchdog DEAD_ERRORS,resident 会被自愈重起)。兜底不吞它。
    fc = _FakeClient(connect_error=True)
    monkeypatch.setattr(vm, "_get_client", lambda: fc)
    snap = await vm.fetch_instance("moss_transcribe_diarize", 8003)
    assert snap.healthy is False
    assert snap.error == "ConnectError"
