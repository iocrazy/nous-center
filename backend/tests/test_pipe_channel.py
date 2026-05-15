"""Lane C: PipeChannel 测试 —— F1 约束（Pipe 不可 await、send 无 timeout）的封装.

用 multiprocessing.Pipe() 在同进程内开一对 conn，两端各包一个 PipeChannel,
不起子进程也能压完整的 asyncio 桥接 + 写超时逻辑。
"""
import asyncio
import multiprocessing as mp

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel, PipeWriteTimeout


@pytest.mark.asyncio
async def test_send_and_recv_round_trip():
    """一端 send_message，另一端 recv_message 拿到等价消息。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        msg = P.RunNode(
            task_id=1, node_id="n", node_type="image",
            model_key="m", inputs={"k": "v"},
        )
        await ch_a.send_message(msg)
        got = await ch_b.recv_message()
        assert got == msg
    finally:
        ch_a.close()
        ch_b.close()


@pytest.mark.asyncio
async def test_recv_eof_raises_connection_closed():
    """对端 close 后，recv_message 抛 ConnectionClosed（runner crash 检测靠它）。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        ch_a.close()  # 模拟对端（runner）崩溃
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(ch_b.recv_message(), timeout=2.0)
    finally:
        ch_b.close()


@pytest.mark.asyncio
async def test_send_times_out_on_slow_consumer():
    """对端不读 pipe，缓冲区写满后 send 应在 write_timeout 内抛 PipeWriteTimeout.

    F1：Pipe.send 本身无 timeout —— PipeChannel 用写线程 + join 超时实现。
    """
    a, b = mp.Pipe()
    # b 端永不读 —— a 端持续 send 直到 OS pipe 缓冲写满后阻塞
    ch_a = PipeChannel(a, write_timeout=2.0)
    try:
        big = P.RunNode(
            task_id=1, node_id="n", node_type="image", model_key="m",
            inputs={"blob": "x" * 100_000},  # 大负载，加速填满缓冲
        )
        with pytest.raises(PipeWriteTimeout):
            # 循环 send，缓冲满后某次 send 会超时
            for _ in range(10_000):
                await ch_a.send_message(big)
    finally:
        ch_a.close()
        b.close()


@pytest.mark.asyncio
async def test_concurrent_sends_are_serialized():
    """多个协程并发 send，写线程串行化，对端收齐所有消息且不交错损坏。"""
    a, b = mp.Pipe()
    ch_a = PipeChannel(a)
    ch_b = PipeChannel(b)
    try:
        n = 20
        await asyncio.gather(*(
            ch_a.send_message(P.NodeProgress(task_id=i, node_id="n", progress=0.5))
            for i in range(n)
        ))
        seen = set()
        for _ in range(n):
            msg = await asyncio.wait_for(ch_b.recv_message(), timeout=2.0)
            seen.add(msg.task_id)
        assert seen == set(range(n))
    finally:
        ch_a.close()
        ch_b.close()
