"""RunnerSupervisor —— 主进程侧的 runner 子进程监管（spec §4.2）.

职责：
  * start()  —— fork runner 子进程，建 RunnerClient，等 Ready 握手。
  * watchdog —— 每 ping_interval 发一次 ping；ping_timeout 内无 Pong 或 pipe
    EOF → 判定 crash → _restart()。
  * _restart() —— 终结旧 runner（terminate → kill）；inflight task 全标 failed
    (runner_crashed)；按 RESTART_BACKOFF 退避；过 F2 GPU-free gate（轮询
    gpu_free_probe 直到该 group 的 GPU 显存回落）；重新 fork + 握手。
  * 成功跑满 stable_seconds（默认 30min）后 reset restart_count（spec §4.2 第 6 步）。

本 Lane 用 fake runner（adapter_class 默认 FakeAdapter）+ 注入式 gpu_free_probe
（不碰真 nvidia-smi）。生产环境 Lane H 接真探针 + resident preload。
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from typing import Callable

from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main

logger = logging.getLogger(__name__)

# spec §4.2 默认值
DEFAULT_PING_INTERVAL = 30.0
DEFAULT_PING_TIMEOUT = 10.0
DEFAULT_RESTART_BACKOFF = [5.0, 15.0, 60.0, 300.0]  # 封顶 5 min
DEFAULT_STABLE_SECONDS = 30 * 60  # 跑满 30min 视为稳定，reset crash count

_SPAWN = mp.get_context("spawn")  # CUDA 子进程惯例：spawn 不 fork


def _default_gpu_free_probe(gpus: list[int]) -> bool:
    """生产用 GPU-free 探针：nvidia-smi 查这些 GPU 的显存是否回落到基线.

    本 Lane 测试注入 fake 探针；这个默认实现是 Lane H 接 resident preload 时
    会用到的真探针骨架 —— 此处保守返回 True（无 GPU 环境不阻塞）。
    """
    return True


class RunnerSupervisor:
    def __init__(
        self,
        *,
        group_id: str,
        gpus: list[int],
        adapter_class: str = "src.runner.fake_adapter.FakeAdapter",
        ping_interval: float = DEFAULT_PING_INTERVAL,
        ping_timeout: float = DEFAULT_PING_TIMEOUT,
        restart_backoff: list[float] | None = None,
        stable_seconds: float = DEFAULT_STABLE_SECONDS,
        gpu_free_probe: Callable[[list[int]], bool] | None = None,
        gpu_free_poll_interval: float = 2.0,
        on_task_failed: Callable[[int, str], None] | None = None,
    ) -> None:
        self.group_id = group_id
        self.gpus = gpus
        self.adapter_class = adapter_class
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.restart_backoff = restart_backoff or DEFAULT_RESTART_BACKOFF
        self.stable_seconds = stable_seconds
        self._gpu_free_probe = gpu_free_probe or _default_gpu_free_probe
        self._gpu_free_poll_interval = gpu_free_poll_interval
        self._on_task_failed = on_task_failed

        self._process: mp.Process | None = None
        self.client: RunnerClient | None = None
        self.restart_count = 0
        self._inflight: set[int] = set()
        self._last_spawn_at = 0.0
        self._watchdog_task: asyncio.Task | None = None
        self._stopping = False
        self._restarted_count = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.is_alive()
            and self.client is not None
            and self.client.is_connected
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def backoff_for(self, restart_index: int) -> float:
        """第 restart_index 次重启该等多久（封顶 restart_backoff 最后一个值）。"""
        if restart_index < len(self.restart_backoff):
            return self.restart_backoff[restart_index]
        return self.restart_backoff[-1]

    # ------------------------------------------------------------------
    # inflight task 登记（crash 时全标 failed）
    # ------------------------------------------------------------------

    def register_inflight(self, task_id: int) -> None:
        self._inflight.add(task_id)

    def unregister_inflight(self, task_id: int) -> None:
        self._inflight.discard(task_id)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._spawn()
        self._watchdog_task = asyncio.create_task(
            self._watchdog(), name=f"watchdog-{self.group_id}"
        )

    async def _spawn(self) -> None:
        """fork runner 子进程 + 建 client + 等 Ready。"""
        parent_conn, child_conn = _SPAWN.Pipe()
        proc = _SPAWN.Process(
            target=runner_main,
            args=(self.group_id, self.gpus, child_conn),
            kwargs={"adapter_class": self.adapter_class},
            daemon=True,
            name=f"runner-{self.group_id}",
        )
        proc.start()
        child_conn.close()  # 主进程侧不用 child 端
        self._process = proc
        self.client = RunnerClient(parent_conn, runner_id=f"runner-{self.group_id}")
        await self.client.start()  # 等 Ready 握手
        self._last_spawn_at = time.monotonic()
        logger.info(
            "runner %s spawned (pid=%s, gpus=%s)", self.group_id, proc.pid, self.gpus
        )

    async def stop(self) -> None:
        """优雅停止 —— 不再重启，终结子进程。"""
        self._stopping = True
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        if self.client is not None:
            await self.client.close()
        await self._terminate_process()

    async def _terminate_process(self) -> None:
        """SIGTERM 5s → SIGKILL。"""
        proc = self._process
        if proc is None:
            return
        if proc.is_alive():
            proc.terminate()
            for _ in range(50):  # 最多等 5s
                if not proc.is_alive():
                    break
                await asyncio.sleep(0.1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3.0)
        else:
            proc.join(timeout=1.0)

    # ------------------------------------------------------------------
    # watchdog + restart
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        """每 ping_interval ping 一次；超时 / EOF → crash → 重启。"""
        while not self._stopping:
            await asyncio.sleep(self.ping_interval)
            if self._stopping:
                return
            try:
                await asyncio.wait_for(self.client.ping(), timeout=self.ping_timeout)
            except (asyncio.TimeoutError, ConnectionError):
                if self._stopping:
                    return
                logger.warning("runner %s ping failed -> restarting", self.group_id)
                await self._restart()

    async def _restart(self) -> None:
        """spec §4.2 crash 检测 + 重启 6 步。"""
        # 1. 终结旧 runner
        await self._terminate_process()
        if self.client is not None:
            await self.client.close()

        # 2. inflight task 全标 failed (runner_crashed)，不重试
        for task_id in list(self._inflight):
            if self._on_task_failed is not None:
                self._on_task_failed(task_id, "runner_crashed")
        self._inflight.clear()

        # 3. backoff（防 crash 风暴）
        backoff = self.backoff_for(self.restart_count)
        await asyncio.sleep(backoff)

        # 4. F2 GPU-free gate —— 轮询直到该 group 的 GPU 显存回落
        while not self._stopping and not self._gpu_free_probe(self.gpus):
            logger.info(
                "runner %s GPU-free gate: GPUs %s not yet free, waiting",
                self.group_id, self.gpus,
            )
            await asyncio.sleep(self._gpu_free_poll_interval)
        if self._stopping:
            return

        # 5. 重新 fork + 握手
        try:
            await self._spawn()
        except Exception:
            logger.exception("runner %s respawn failed", self.group_id)
            self.restart_count += 1
            return

        # 6. restart_count 累加；成功跑满 stable_seconds 后由
        #    _maybe_reset_crash_count reset（生产由 Lane H 健康检查 loop 调）
        self.restart_count += 1
        self._restarted_count += 1
        logger.info(
            "runner %s restarted (count=%d, backoff=%.1fs)",
            self.group_id, self.restart_count, backoff,
        )

    def _maybe_reset_crash_count(self) -> None:
        """跑满 stable_seconds 无 crash → reset restart_count（spec §4.2 第 6 步）.

        由调用方（Lane H 的健康检查 loop）周期性调用，或 watchdog 每轮调。
        """
        if (
            self.restart_count > 0
            and self._last_spawn_at > 0
            and time.monotonic() - self._last_spawn_at >= self.stable_seconds
        ):
            logger.info(
                "runner %s stable for %.0fs -> reset crash count",
                self.group_id, self.stable_seconds,
            )
            self.restart_count = 0

    # ------------------------------------------------------------------
    # 测试辅助
    # ------------------------------------------------------------------

    async def wait_restarted(self, count: int = 1) -> None:
        """等到 supervisor 完成至少 count 次重启（测试用）。"""
        while self._restarted_count < count:
            await asyncio.sleep(0.05)
