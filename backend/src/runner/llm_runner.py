"""LLMRunner —— vLLM 子进程生命周期管理（spec §1.2 / §4.1 / §4.2）。

与 image/TTS 的 RunnerSupervisor（Lane C）本质不同：
  * RunnerSupervisor fork 一个 multiprocessing.Process 跑 runner_main（内有
    pipe-reader/executor 双 task + per-group PriorityQueue），串行化 GPU job。
  * LLMRunner 不 fork Python 子进程 —— 它是主进程内的对象，管的「子进程」是
    vLLM 本身（VLLMAdapter 启的 subprocess.Popen）。LLMRunner 无 PriorityQueue、
    无 IPC pipe、不串行化推理 —— 推理请求由 compat 路由 / executor 直连
    vLLM HTTP（spec §4.5 D6/D8），并发由 vLLM continuous batching 处理（spec §1.3）。

职责（spec 用词）：
  * spawn()    —— 触发 VLLMAdapter.load()，起 vLLM 子进程 + 等 health。
  * health()   —— 探测 vLLM 子进程是否存活（VLLMAdapter._health_check）。
  * preload()  —— resident LLM 启动加载，fail-soft（失败不抛，记 failure，
                  对齐 spec §4.2「load_failed 不阻断 API server start」）。
  * abort()    —— 向 vLLM 发 HTTP abort（within-node cancel，spec §2.2）。
  * restart()  —— vLLM crash / OOM 退出 → kill orphan → 过 F2 GPU-free gate
                  → re-spawn → re-preload（spec §4.1「LLM Runner crash → 重启
                  runner → re-spawn vLLM + preload」）。
"""
from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LLMRunnerState(str, enum.Enum):
    IDLE = "idle"          # 尚未 spawn
    RUNNING = "running"    # vLLM 子进程健康
    FAILED = "failed"      # spawn / preload 失败（fail-soft，failure 字段有原因）
    RESTARTING = "restarting"


def _default_gpu_free_probe(gpus: list[int]) -> bool:
    """生产用 F2 GPU-free 探针骨架：查 role:llm group 的 GPU 显存是否回落。

    死进程的 CUDA context 回收是异步的（spec §4.2 F2）—— re-spawn vLLM 前必须
    确认显存已释放，否则新 vLLM 立刻 OOM。本默认实现查 nvidia-smi；无 GPU 环境
    （CI / CUDA_VISIBLE_DEVICES=""）保守返回 True 不阻塞。测试注入 fake 探针。
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return True  # 无 nvidia-smi → 不阻塞
        # 该 group 任意 GPU used > 2GB 视为 context 未回收（保守阈值）。
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) in gpus:
                if int(parts[1]) > 2048:
                    return False
        return True
    except Exception:
        return True  # 探针本身出错 → 不阻塞（保守，避免永久卡住重启）


class LLMRunner:
    """主进程内的 vLLM 子进程生命周期管理对象。不串行化推理请求。"""

    def __init__(
        self,
        *,
        model_key: str,
        adapter: Any,
        llm_gpus: list[int],
        gpu_free_probe: Callable[[list[int]], bool] | None = None,
        gpu_free_poll_interval: float = 2.0,
        gpu_free_max_wait: float = 120.0,
    ) -> None:
        self.model_key = model_key
        self.adapter = adapter           # VLLMAdapter（或测试的 FakeVLLMAdapter）
        self.llm_gpus = llm_gpus         # role:llm group 的 GPU index（Lane A allocator.llm_group_gpus()）
        self.state = LLMRunnerState.IDLE
        self.failure: str | None = None
        self._gpu_free_probe = gpu_free_probe or _default_gpu_free_probe
        self._gpu_free_poll_interval = gpu_free_poll_interval
        self._gpu_free_max_wait = gpu_free_max_wait
        # 注意：刻意不持有 PriorityQueue / inflight dict —— LLMRunner 不串行化推理。

    @property
    def base_url(self) -> str | None:
        """vLLM HTTP 端点 —— compat 路由 / executor 直连用（经 get_vllm_base_url）。"""
        return getattr(self.adapter, "base_url", None)

    async def spawn(self) -> None:
        """起 vLLM 子进程 + 等 health。失败抛 —— 调用方（preload）决定是否 fail-soft。"""
        await self.adapter.load()
        self.state = LLMRunnerState.RUNNING
        self.failure = None
        logger.info("LLMRunner %s spawned (base_url=%s)", self.model_key, self.base_url)

    async def health(self) -> bool:
        """探测 vLLM 子进程是否存活。并发安全 —— 无锁、不阻塞推理路径。"""
        try:
            return await self.adapter._health_check()
        except Exception:
            return False

    async def preload(self) -> None:
        """resident LLM 启动加载 —— fail-soft：失败不抛，记 failure（spec §4.2）。"""
        try:
            await self.spawn()
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            self.state = LLMRunnerState.FAILED
            self.failure = detail
            logger.warning("LLMRunner %s preload failed: %s", self.model_key, detail)

    async def abort(self, request_id: str) -> None:
        """向 vLLM 发 HTTP abort（within-node LLM cancel，spec §2.2）。

        vLLM 的 OpenAI-compat server 不暴露按 request_id 的 abort 端点；实际
        within-node cancel 由 compat 路由 / executor 关闭 httpx stream 实现
        （spec §4.4「LLM streaming: cancel_event → vllm_http_abort → 关流」）。
        本方法是 LLMRunner 侧的抽象入口 —— 当前为 best-effort no-op + 日志，
        真正的取消语义在 streaming 调用方那一侧（关闭连接 = vLLM 感知 disconnect
        并停止该序列的 decode）。保留此方法是为接口完整 + 未来 vLLM 暴露 abort
        端点时的挂载点。
        """
        logger.debug("LLMRunner %s abort request_id=%s (handled by stream close)",
                     self.model_key, request_id)

    async def _wait_gpu_free(self) -> None:
        """F2 GPU-free gate：轮询探针直到 role:llm group 的 GPU 显存回落。"""
        waited = 0.0
        while waited < self._gpu_free_max_wait:
            if self._gpu_free_probe(self.llm_gpus):
                return
            await asyncio.sleep(self._gpu_free_poll_interval)
            waited += self._gpu_free_poll_interval
        logger.warning(
            "LLMRunner %s GPU-free gate 超时（%.0fs）—— 仍尝试 re-spawn",
            self.model_key, self._gpu_free_max_wait,
        )

    async def restart(self) -> None:
        """vLLM crash / OOM 退出 → kill orphan → GPU-free gate → re-spawn → re-preload。

        spec §4.1：「LLM Runner crash → vLLM 也随之失联 → 重启 runner → 重新
        spawn vLLM + preload」。
        """
        self.state = LLMRunnerState.RESTARTING
        logger.warning("LLMRunner %s restarting (vLLM crash/OOM detected)", self.model_key)

        # 1. kill 残留 orphan（VLLMAdapter.unload 内部走 _kill_process，
        #    SIGTERM 进程组 → SIGKILL；adopted orphan 也覆盖）。
        try:
            self.adapter.unload()
        except Exception as e:
            logger.warning("LLMRunner %s unload during restart failed: %s",
                           self.model_key, e)

        # 2. F2 GPU-free gate —— 等死进程的 CUDA context 回收。
        await self._wait_gpu_free()

        # 3. re-spawn + re-preload（fail-soft）。
        await self.preload()

    async def shutdown(self) -> None:
        """优雅停止 —— 终结 vLLM 子进程。"""
        try:
            self.adapter.unload()
        finally:
            self.state = LLMRunnerState.IDLE
