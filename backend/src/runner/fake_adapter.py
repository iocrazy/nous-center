"""FakeAdapter —— 零 GPU / 零模型的 InferenceAdapter 实现.

V1.5 Lane C 用它跑通 runner 子进程框架（IPC + 生命周期）而不需要真硬件。
Lane D/E/F 把真 adapter（image_diffusers / TTS / vLLM）迁进 runner 时，复用
本 Lane 验证过的同一套 runner / supervisor 代码路径。

可配置的行为开关（构造参数）：
  fail_load        —— load() 抛 FakeLoadError（模拟权重文件丢失 / OOM）
  crash_on_infer   —— infer() 抛 RuntimeError（模拟节点执行期 native fault）
  infer_seconds    —— 每个 step 的模拟耗时（asyncio.sleep，可让出 event loop）

infer() 支持 progress_callback（每 step 回调一次）和 cancel_flag（threading.Event,
set 后下一 step 边界抛 asyncio.CancelledError）—— 形状对齐真 image adapter 的
diffusers callback_on_step_end + CancelFlag（spec §4.4）。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, ClassVar

from src.services.inference.base import (
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


class FakeLoadError(Exception):
    """FakeAdapter.load() 在 fail_load=True 时抛 —— 模拟加载失败。"""


class FakeOOMError(RuntimeError):
    """FakeAdapter.load() 在 oom_on_load_count 未耗尽时抛 —— 模拟 CUDA OOM。

    类名刻意含 'OutOfMemoryError' 子串，让 ModelManager._is_oom 据类名判定它
    走 OOM-evict-retry 路径（不能依赖 torch.cuda.OutOfMemoryError —— 测试里
    torch 是 MagicMock）。
    """


class FakeAdapter(InferenceAdapter):
    """假 adapter：可配置 crash / slow / fail-load / 多 step 进度。"""

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    estimated_vram_mb: ClassVar[int] = 0

    def __init__(
        self,
        paths: dict[str, str],
        device: str = "cpu",
        *,
        fail_load: bool = False,
        crash_on_infer: bool = False,
        infer_seconds: float = 0.01,
        oom_on_load_count: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(paths, device, **params)
        self._fail_load = fail_load
        self._crash_on_infer = crash_on_infer
        self._infer_seconds = infer_seconds
        self._oom_on_load_count = oom_on_load_count
        self._load_attempts = 0

    async def load(self, device: str) -> None:
        await asyncio.sleep(0)  # 可让出
        self._load_attempts += 1
        if self._load_attempts <= self._oom_on_load_count:
            raise FakeOOMError(
                f"CUDA out of memory (fake, attempt {self._load_attempts})"
            )
        if self._fail_load:
            raise FakeLoadError(f"fake load failure for paths={self.paths}")
        self.device = device
        self._model = object()  # 非 None → is_loaded True

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> InferenceResult:
        if self._crash_on_infer:
            raise RuntimeError("fake adapter crash during infer")

        steps = int(getattr(req, "steps", 1) or 1)
        started = time.monotonic()
        for done in range(1, steps + 1):
            # 每 step 边界检查 cancel —— 对齐真 adapter 的 callback_on_step_end
            if cancel_flag is not None and cancel_flag.is_set():
                raise asyncio.CancelledError()
            if self._infer_seconds > 0:
                await asyncio.sleep(self._infer_seconds)  # 可让出 event loop
            else:
                await asyncio.sleep(0)
            if progress_callback is not None:
                progress_callback(done, steps)

        latency_ms = int((time.monotonic() - started) * 1000)
        return InferenceResult(
            media_type="image/png",
            data=b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes",
            metadata={"fake": True, "steps": steps},
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
