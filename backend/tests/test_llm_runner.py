"""Lane E: LLMRunner 测试 —— vLLM 子进程生命周期，不串行化推理。

全程用 FakeVLLMAdapter（不起真 vLLM、不碰 GPU），注入式 gpu_free_probe。
"""
import asyncio

import pytest

from src.runner.llm_runner import LLMRunner, LLMRunnerState


class FakeVLLMAdapter:
    """模拟 VLLMAdapter 的生命周期接口子集，零子进程零 GPU。"""

    def __init__(self, *, fail_load: bool = False, base_url: str = "http://localhost:8123"):
        self._fail_load = fail_load
        self.base_url = base_url
        self.is_loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self._alive = False  # 模拟 vLLM 子进程存活

    async def load(self, device=None):
        self.load_calls += 1
        await asyncio.sleep(0)  # 可让出
        if self._fail_load:
            raise RuntimeError("fake vLLM failed to start: OOM")
        self.is_loaded = True
        self._alive = True

    def unload(self):
        self.unload_calls += 1
        self.is_loaded = False
        self._alive = False

    async def _health_check(self) -> bool:
        return self._alive

    def simulate_crash(self):
        """模拟 vLLM 子进程 OOM / crash 退出。"""
        self._alive = False
        self.is_loaded = False


@pytest.mark.asyncio
async def test_spawn_loads_vllm_subprocess():
    """spawn() → 调 adapter.load() → state=running。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    assert adapter.load_calls == 1
    assert adapter.is_loaded
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_health_returns_true_when_vllm_alive():
    """health() 反映 vLLM 子进程存活状态。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    assert await runner.health() is True
    adapter.simulate_crash()
    assert await runner.health() is False


@pytest.mark.asyncio
async def test_preload_failsoft_records_failure():
    """preload 时 vLLM 启动失败 → fail-soft：不抛，state=failed，failure 可读。"""
    adapter = FakeVLLMAdapter(fail_load=True)
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.preload()  # 不抛
    assert runner.state == LLMRunnerState.FAILED
    assert "OOM" in (runner.failure or "")


@pytest.mark.asyncio
async def test_preload_success():
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.preload()
    assert runner.state == LLMRunnerState.RUNNING
    assert runner.failure is None


@pytest.mark.asyncio
async def test_restart_on_crash_respawns_and_passes_gpu_free_gate():
    """vLLM crash → restart()：kill orphan → 过 GPU-free gate → re-spawn。"""
    adapter = FakeVLLMAdapter()
    gate_calls: list[list[int]] = []

    def fake_gpu_free_probe(gpus):
        gate_calls.append(list(gpus))
        return True  # 显存已回落

    runner = LLMRunner(
        model_key="qwen", adapter=adapter, llm_gpus=[0, 1],
        gpu_free_probe=fake_gpu_free_probe, gpu_free_poll_interval=0.01,
    )
    await runner.spawn()
    adapter.simulate_crash()
    await runner.restart()
    assert adapter.unload_calls >= 1          # kill orphan
    assert gate_calls == [[0, 1]]             # GPU-free gate 查的是 llm group GPU
    assert adapter.load_calls == 2            # re-spawn
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_restart_blocked_until_gpu_free_gate_passes():
    """GPU-free gate 探针先返回 False（CUDA context 未回收）→ restart 卡住，
    回落后才 re-spawn（F2）。"""
    adapter = FakeVLLMAdapter()
    probe_results = iter([False, False, True])

    def fake_probe(gpus):
        return next(probe_results, True)

    runner = LLMRunner(
        model_key="qwen", adapter=adapter, llm_gpus=[0, 1],
        gpu_free_probe=fake_probe, gpu_free_poll_interval=0.01,
    )
    await runner.spawn()
    adapter.simulate_crash()
    await runner.restart()
    # 探针前两次 False → 等待；第三次 True → 继续 re-spawn
    assert adapter.load_calls == 2
    assert runner.state == LLMRunnerState.RUNNING


@pytest.mark.asyncio
async def test_abort_does_not_serialize_requests():
    """关键不变量：LLMRunner 不持有队列、不串行化。abort 是对 vLLM 的 HTTP 信号，
    不阻塞其它推理。本测试断言 LLMRunner 没有 PriorityQueue / inflight 串行化结构。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    # LLMRunner 不该有任何串行化推理的队列结构
    assert not hasattr(runner, "queue")
    assert not hasattr(runner, "_priority_queue")
    # abort 接口存在且可调（fake 下是 no-op，真实现发 vLLM HTTP abort）
    await runner.abort(request_id="req-123")  # 不抛


@pytest.mark.asyncio
async def test_concurrent_health_checks_do_not_block():
    """并发调 health() 不互相阻塞 —— LLMRunner 无串行化锁卡住推理路径。"""
    adapter = FakeVLLMAdapter()
    runner = LLMRunner(model_key="qwen", adapter=adapter, llm_gpus=[0, 1])
    await runner.spawn()
    results = await asyncio.gather(*[runner.health() for _ in range(10)])
    assert all(results)
