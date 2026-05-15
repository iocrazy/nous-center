"""PipeChannel —— multiprocessing.Pipe 的 asyncio 桥接（F1 约束封装）。

spec §3.3 F1：
  * `multiprocessing.Pipe` 对象不可直接 await —— 读侧把底层 fd 注册进 event loop
    (`loop.connect_read_pipe`)，拿不到 fd 时退化为读线程桥到 asyncio.Queue。
  * `Pipe.send` 无 timeout 参数 —— 写侧用一个专用写线程 + queue.Queue，对每次
    send 加超时（默认 5s，spec §4.1）。send 超时 = 对端假死，由上层（supervisor）
    判定为 runner crash。

wire format：每条消息 = 4-byte big-endian 长度前缀 + protocol.encode 出的 body。
长度前缀让读侧能在字节流上切出完整 frame（connect_read_pipe 给的是字节流，不是
multiprocessing 的对象边界）。
"""
from __future__ import annotations

import asyncio
import os
import queue
import struct
import threading
from typing import Any

from src.runner import protocol as P

# 默认写超时（秒），spec §4.1「IPC pipe 阻塞 5s 超时」
DEFAULT_WRITE_TIMEOUT = 5.0
# 4-byte 长度前缀
_LEN_PREFIX = struct.Struct(">I")


class PipeWriteTimeout(Exception):
    """send 在 write_timeout 内未完成 —— 对端假死。"""


class _LengthPrefixedProtocol(asyncio.Protocol):
    """connect_read_pipe 用的 asyncio.Protocol：在字节流上切 length-prefixed frame，
    完整 frame 解码后丢进 asyncio.Queue。"""

    def __init__(self, out_queue: asyncio.Queue, fmt: str) -> None:
        self._queue = out_queue
        self._fmt = fmt
        self._buf = bytearray()
        self._eof = False

    def data_received(self, data: bytes) -> None:
        self._buf.extend(data)
        while len(self._buf) >= _LEN_PREFIX.size:
            (body_len,) = _LEN_PREFIX.unpack_from(self._buf, 0)
            if len(self._buf) < _LEN_PREFIX.size + body_len:
                break  # frame 还没收全
            start = _LEN_PREFIX.size
            body = bytes(self._buf[start:start + body_len])
            del self._buf[:start + body_len]
            try:
                msg = P.decode(body, fmt=self._fmt)
            except P.ProtocolError as e:
                self._queue.put_nowait(_DecodeFailure(e))
                continue
            self._queue.put_nowait(msg)

    def eof_received(self) -> None:
        self._eof = True
        self._queue.put_nowait(_EOF)

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._eof:
            self._queue.put_nowait(_EOF)


class _DecodeFailure:
    def __init__(self, err: P.ProtocolError) -> None:
        self.err = err


_EOF = object()  # sentinel：对端 close / EOF


class PipeChannel:
    """一端 multiprocessing.Pipe connection 的 asyncio 包装。

    读：`recv_message()` —— event loop 友好，永不阻塞 loop。
    写：`send_message()` —— 经专用写线程，对 send 加 write_timeout。
    """

    def __init__(
        self,
        conn: Any,
        *,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        fmt: str | None = None,
    ) -> None:
        self._conn = conn
        self._write_timeout = write_timeout
        self._fmt = fmt or P.default_format()
        self._closed = False

        # —— 读侧 ——
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._transport: asyncio.ReadTransport | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_started = False

        # —— 写侧 ——
        # 每个待写 frame = (bytes, threading.Event done, list[Exception|None] err)
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="pipe-writer", daemon=True
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # 读侧
    # ------------------------------------------------------------------

    async def _ensure_reader(self) -> None:
        """惰性启动读侧 —— 优先 connect_read_pipe，失败退化为读线程桥。"""
        if self._reader_started:
            return
        self._reader_started = True
        loop = asyncio.get_running_loop()
        try:
            # connect_read_pipe 需要一个有 fileno() 的可读对象。
            # multiprocessing.Connection 在 POSIX 上 fileno() 返回底层 fd。
            pipe_obj = os.fdopen(os.dup(self._conn.fileno()), "rb", buffering=0)
            transport, _ = await loop.connect_read_pipe(
                lambda: _LengthPrefixedProtocol(self._recv_queue, self._fmt),
                pipe_obj,
            )
            self._transport = transport
        except (OSError, ValueError, NotImplementedError):
            # 拿不到 fd（Windows / 特殊平台）→ 退化为读线程桥到 asyncio.Queue
            self._start_reader_thread(loop)

    def _start_reader_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        def _loop() -> None:
            while not self._closed:
                try:
                    msg = self._conn.recv()  # 阻塞读一个 multiprocessing 对象
                except EOFError:
                    loop.call_soon_threadsafe(self._recv_queue.put_nowait, _EOF)
                    return
                except OSError:
                    loop.call_soon_threadsafe(self._recv_queue.put_nowait, _EOF)
                    return
                loop.call_soon_threadsafe(self._recv_queue.put_nowait, msg)

        self._reader_thread = threading.Thread(
            target=_loop, name="pipe-reader-bridge", daemon=True
        )
        self._reader_thread.start()

    async def recv_message(self) -> Any:
        """收一条消息。对端 close → 抛 ConnectionError。解码失败 → 抛 ProtocolError。"""
        await self._ensure_reader()
        item = await self._recv_queue.get()
        if item is _EOF:
            raise ConnectionError("pipe peer closed (EOF)")
        if isinstance(item, _DecodeFailure):
            raise item.err
        return item

    # ------------------------------------------------------------------
    # 写侧
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """专用写线程：从 _write_queue 取 frame，写 pipe，结果回填 Event。"""
        while True:
            item = self._write_queue.get()
            if item is None:  # close 信号
                return
            body, done, err_box = item
            try:
                # connect_read_pipe 读侧期望 length-prefixed 字节流 →
                # 用 send_bytes 写裸字节（不是 send 的 pickle 协议）。
                self._conn.send_bytes(body)
            except Exception as e:  # noqa: BLE001
                err_box[0] = e
            finally:
                done.set()

    async def send_message(self, msg: Any) -> None:
        """发一条消息。write_timeout 内未完成 → 抛 PipeWriteTimeout。"""
        if self._closed:
            raise ConnectionError("channel closed")
        body = P.encode(msg, fmt=self._fmt)
        # NOTE: multiprocessing.Connection.send_bytes 在底层 fd 上写
        # 「4-byte big-endian 长度前缀 + body」—— 所以 wire format 天然
        # 就是 length-prefixed，不需要我们再加前缀。_LengthPrefixedProtocol
        # 的 data_received 就是切这个 send_bytes 帧。
        done = threading.Event()
        err_box: list[Exception | None] = [None]
        self._write_queue.put((body, done, err_box))
        loop = asyncio.get_running_loop()
        # 在 executor 里等 Event，不阻塞 event loop
        finished = await loop.run_in_executor(
            None, done.wait, self._write_timeout
        )
        if not finished:
            raise PipeWriteTimeout(
                f"pipe send did not complete in {self._write_timeout}s"
            )
        if err_box[0] is not None:
            raise ConnectionError(f"pipe send failed: {err_box[0]}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_queue.put(None)  # 停写线程
        if self._transport is not None:
            self._transport.close()
        try:
            self._conn.close()
        except OSError:
            pass
