"""Lane C: RunnerSupervisor 测试 —— spawn / watchdog / crash 重启 / GPU-free gate.

用 fake runner 子进程 + 注入的 GPU-free 探针（不碰 nvidia-smi）。
"""
import asyncio

import pytest

from src.runner import protocol as P
from src.runner.supervisor import RunnerSupervisor


def _make_supervisor(**overrides) -> RunnerSupervisor:
    """构造一个跑 fake runner 的 supervisor，超时参数缩小以便快测。"""
    kw = dict(
        group_id="image",
        gpus=[2],
        adapter_class="src.runner.fake_adapter.FakeAdapter",
        ping_interval=0.3,
        ping_timeout=0.5,
        restart_backoff=[0.1, 0.2, 0.3],
        gpu_free_probe=lambda gpus: True,  # 默认 GPU 立即 free
    )
    kw.update(overrides)
    return RunnerSupervisor(**kw)


@pytest.mark.asyncio
async def test_start_spawns_runner_and_handshakes():
    sup = _make_supervisor()
    try:
        await sup.start()
        assert sup.is_running
        assert sup.client.is_ready
        # 能正常派活
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node(P.RunNode(
            task_id=1, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 2},
        ))
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_watchdog_detects_crash_and_restarts():
    """杀掉 runner 子进程 → watchdog ping 超时 → 自动重启 → 新 runner 可用。"""
    sup = _make_supervisor()
    try:
        await sup.start()
        old_pid = sup.pid
        # 模拟 crash
        sup._process.terminate()
        # 等 watchdog 检测 + 重启（ping_interval + ping_timeout + backoff + 重启）
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sup.is_running
        assert sup.pid != old_pid  # 新进程
        assert sup.restart_count == 1
        # 新 runner 能干活
        await sup.client.load_model("fake-img", config={})
        result = await sup.client.run_node(P.RunNode(
            task_id=2, node_id="n", node_type="image",
            model_key="fake-img", inputs={"steps": 2},
        ))
        assert result.status == "completed"
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_crash_marks_inflight_tasks_failed():
    """runner crash 时，supervisor 把登记的 inflight task 全标 failed（runner_crashed）。"""
    failed: list[tuple[int, str]] = []
    sup = _make_supervisor(
        on_task_failed=lambda task_id, reason: failed.append((task_id, reason)),
    )
    try:
        await sup.start()
        # 登记两个 inflight task
        sup.register_inflight(101)
        sup.register_inflight(102)
        sup._process.terminate()
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sorted(t for t, _ in failed) == [101, 102]
        assert all(reason == "runner_crashed" for _, reason in failed)
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_restart_backoff_sequence():
    """连续 crash —— backoff 按 restart_backoff 序列递增，封顶最后一个值。"""
    sup = _make_supervisor(restart_backoff=[0.1, 0.3, 0.5])
    try:
        await sup.start()
        # backoff_for(n) 给第 n 次重启该等多久
        assert sup.backoff_for(0) == 0.1
        assert sup.backoff_for(1) == 0.3
        assert sup.backoff_for(2) == 0.5
        assert sup.backoff_for(3) == 0.5  # 封顶
        assert sup.backoff_for(99) == 0.5
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_gpu_free_gate_blocks_restart_until_clear():
    """GPU-free gate：探针返回 False 时重启被卡住，返回 True 后才继续（F2）。"""
    gate_state = {"free": False}
    probe_calls: list[int] = []

    def _probe(gpus):
        probe_calls.append(1)
        return gate_state["free"]

    sup = _make_supervisor(gpu_free_probe=_probe, gpu_free_poll_interval=0.1)
    try:
        await sup.start()
        sup._process.terminate()
        # gate 卡住 —— 0.5s 内不应完成重启
        await asyncio.sleep(1.0)
        assert not sup.is_running or sup.restart_count == 0
        assert len(probe_calls) >= 2  # gate 在轮询
        # 放行
        gate_state["free"] = True
        await asyncio.wait_for(sup.wait_restarted(count=1), timeout=15.0)
        assert sup.is_running
        assert sup.restart_count == 1
    finally:
        await sup.stop()
