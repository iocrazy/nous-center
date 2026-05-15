"""Lane C: RunnerClient 测试 —— 主进程侧节点级 RPC.

用真 multiprocessing fake runner 子进程，验证 RunnerClient 把 pipe 上的消息流
demux 成「per-node 等待 + 进度回调」。

Lane D 后：runner_main 用 ModelManager；kwarg `adapter_class` 替换为
`fake_adapter=True` + `models_yaml_path`，model_key 来自 yaml fixture。
"""
import asyncio
import multiprocessing as mp
from pathlib import Path

import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main

_SPAWN = mp.get_context("spawn")
_FIXTURE = str(Path(__file__).parent / "fixtures" / "runner_models.yaml")


async def _make_client(group_id="image", gpus=(2,)) -> tuple:
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(
        target=runner_main,
        args=(group_id, list(gpus), child_conn),
        kwargs={"models_yaml_path": _FIXTURE, "fake_adapter": True},
        daemon=True,
    )
    proc.start()
    child_conn.close()
    client = RunnerClient(parent_conn, runner_id=f"runner-{group_id}")
    await client.start()  # 起 demux 协程 + 等 Ready
    return proc, client


async def _teardown(proc, client):
    await client.close()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3.0)


@pytest.mark.asyncio
async def test_start_waits_for_ready():
    proc, client = await _make_client(group_id="image", gpus=(2,))
    try:
        assert client.is_ready
        assert client.gpus == [2]
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_load_model_returns_on_model_event():
    proc, client = await _make_client()
    try:
        ok = await client.load_model("fake-img-a", config={})
        assert ok is True
        # fail_load 模型 —— 返回 False
        bad = await client.load_model("fake-img-b", config={"fail_load": True})
        assert bad is False
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_run_node_resolves_with_node_result():
    proc, client = await _make_client()
    try:
        await client.load_model("fake-img-a", config={})
        progress_seen: list[float] = []
        result = await client.run_node(
            P.RunNode(
                task_id=11, node_id="sampler", node_type="image",
                model_key="fake-img-a", inputs={"steps": 4},
            ),
            on_progress=lambda pr: progress_seen.append(pr.progress),
        )
        assert isinstance(result, P.NodeResult)
        assert result.status == "completed"
        assert result.task_id == 11
        assert len(progress_seen) == 4  # 4 step → 4 个 progress 回调
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_ping_returns_pong():
    proc, client = await _make_client()
    try:
        pong = await client.ping()
        assert isinstance(pong, P.Pong)
    finally:
        await _teardown(proc, client)


@pytest.mark.asyncio
async def test_recv_eof_marks_client_disconnected():
    """runner 子进程死掉 → pipe EOF → client 的 inflight run_node 异常结束。"""
    proc, client = await _make_client()
    try:
        await client.load_model("fake-img-a", config={"infer_seconds": 0.2})

        # 跑一个长节点，执行中杀掉 runner
        run_task = asyncio.create_task(client.run_node(P.RunNode(
            task_id=12, node_id="sampler", node_type="image",
            model_key="fake-img-a", inputs={"steps": 50},
        )))
        await asyncio.sleep(0.3)
        proc.terminate()  # 模拟 crash
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(run_task, timeout=5.0)
        assert not client.is_connected
    finally:
        await _teardown(proc, client)
