"""Lane C chaos: pipe 慢消费者 —— F1 写超时 + 反压验证.

spec §5.5 / review 报告 F1：fake runner 不读 pipe → 主进程 PipeChannel 应在
write_timeout 内抛 PipeWriteTimeout（Pipe.send 本身无 timeout，靠写线程实现）.
这是 F1 实现约束的正确性边界压测。
"""
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel, PipeWriteTimeout

_SPAWN = mp.get_context("spawn")


def _silent_child(conn) -> None:
    """一个永不读 pipe 的子进程 —— 模拟假死的 runner。"""
    import time

    time.sleep(30)  # 啥也不干，conn 缓冲很快写满
    conn.close()


@pytest.mark.asyncio
async def test_send_to_silent_runner_times_out():
    """对端进程活着但不读 —— send 填满 OS pipe 缓冲后在 write_timeout 内超时。"""
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(target=_silent_child, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    ch = PipeChannel(parent_conn, write_timeout=3.0)
    try:
        big = P.RunNode(
            task_id=1, node_id="n", node_type="image", model_key="m",
            inputs={"blob": "x" * 200_000},
        )
        with pytest.raises(PipeWriteTimeout):
            for _ in range(100_000):
                await ch.send_message(big)
    finally:
        ch.close()
        proc.terminate()
        proc.join(timeout=3.0)
