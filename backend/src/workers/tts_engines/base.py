from __future__ import annotations

import asyncio
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


class TTSResult(BaseModel):
    """Concrete-engine return shape from synthesize(). TTSEngine.infer wraps
    into the unified InferenceResult envelope before returning to callers."""

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"

    model_config = {"arbitrary_types_allowed": True}


async def _tts_progress_ticker(
    pt: Any, t0: float, est_total_sec: float,
) -> None:
    """PR-4:periodic ticker 改用 ProgressTracker.tick(elapsed, est_total)。
    300ms 间隔由 sleep 控制;tick 内部估 progress / ETA + throttle 兜底。
    被 cancel 时静默退出(infer 的 finally 负责 await task)。"""
    import time

    try:
        while True:
            await asyncio.sleep(0.3)
            elapsed = time.monotonic() - t0
            pt.tick(elapsed, est_total_sec, stage="tts_synth")
    except asyncio.CancelledError:
        return


class TTSEngine(InferenceAdapter):
    modality = MediaModality.AUDIO
    estimated_vram_mb = 0

    def __init__(self, paths: dict[str, str], device: str = "cuda", **params: Any):
        super().__init__(paths=paths, device=device)
        # All TTS engines are single-component: paths['main'] is the model dir.
        self.model_path = Path(paths.get("main", ""))

    async def load(self, device: str | None = None) -> None:
        if device:
            self.device = device
        # Wrap potentially-blocking model load in a thread so the event loop
        # stays responsive during long startup operations.
        await asyncio.to_thread(self.load_sync)

    def load_sync(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        ...

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Any | None = None,
        cancel_flag: Any | None = None,
    ) -> InferenceResult:
        """统一 InferenceAdapter 契约 + PR-1b 任务面板 L3 stage 事件。

        - `progress_callback(done, total, **extras)` 可选 ——
          synthesize 前后各 emit 一帧(`stage="tts_synth"` / progress=0 → 1)。
          每帧带 stage/progress/detail/step_latency_ms/eta_ms,前端 ActiveTaskRow callout 据此
          显示「🔊 合成中 … · ETA」(spec §State model TaskProgress)。
        - `cancel_flag` 可选 —— 节点边界 cancel(spec §4.4 升级:从「TTS 只 boundary-only,
          不接 progress/cancel」放宽为「可选 boundary-cancel + chunk-level 进度」)。
          synthesize 内部仍同步,cancel 只在 to_thread 前后查;真正 cancel-mid-synth
          需各 engine 实现 step-level interrupt(PR-1b 范围外)。

        per-chunk 进度:本基类只发 start/end 两帧;支持 streaming 的具体 engine
        (cosyvoice2 等)在 synthesize 内手动调 progress_callback 多次,基类的兜底
        start/end 仍然成立(`progress` 值从 0→中间值→1 单调,不冲突)。
        """
        if not isinstance(req, AudioRequest):
            raise TypeError(f"TTSEngine expects AudioRequest, got {type(req).__name__}")
        import time

        # PR-4:用 ProgressTracker(throttle 300ms,对齐 PR-1b ticker 间隔)。
        t0 = time.monotonic()
        # 估秒数:经验值 ~0.08 秒/字符(中文密度;英文略快,折中保守)。
        # 实际值在 end 帧由 result.duration_seconds 回填,中间的 ticker 用这个估值算 ETA。
        est_total_sec = max(1.0, len(req.text) * 0.08)
        est_total_units = max(1, int(round(est_total_sec)))
        from src.services.inference.progress_tracker import ProgressTracker  # noqa: PLC0415
        pt = ProgressTracker(progress_callback, stage="tts_synth", throttle_ms=300)
        # start 帧:stage(progress=0)— 让 UI 第一时间显示「正在合成」不黑屏。
        pt.stage("tts_synth", progress=0.0, step=0, total_steps=est_total_units,
                 detail="tts_synth start")

        # spec §4.4 升级:boundary cancel(synthesize 调用前查一次;调用中通过
        # cancel_flag 透传给支持 step-level interrupt 的具体 engine)。这里仅
        # boundary 已经修正了「点 cancel 但 GPU 仍跑完整轮」的老问题。
        if cancel_flag is not None and cancel_flag.is_set():
            raise asyncio.CancelledError()

        # PR-1b/PR-4:periodic ticker —— synthesize 在 to_thread 跑,event loop 同时跑
        # 一个 ticker 每 ~300ms 估算 elapsed/est_total 发 progress 帧(pt.tick 内部
        # 处理 throttle + cap 0.95 + eta)。
        ticker_task = None
        if progress_callback is not None:
            ticker_task = asyncio.create_task(_tts_progress_ticker(pt, t0, est_total_sec))

        try:
            # synthesize() is sync (blocking torch); offload to thread so the
            # event loop stays responsive. Per-engine async-native rewrite is
            # explicitly out of scope (each engine owns blocking torch internals).
            result = await asyncio.to_thread(
                self.synthesize,
                text=req.text,
                voice=req.voice,
                speed=req.speed,
                sample_rate=req.sample_rate,
                reference_audio=req.reference_audio,
                reference_text=req.reference_text,
                emotion=req.emotion,
            )
        finally:
            # synthesize 完成 / 异常 / cancel —— ticker 必须停,避免 leaked task。
            if ticker_task is not None:
                ticker_task.cancel()
                try:
                    await ticker_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        latency_ms = int((time.monotonic() - t0) * 1000)

        # PR-4:end 帧用 pt.finish() — 真实 duration_seconds 回填 total。
        # detail 给前端展示「✓ N.NN 秒」。step=total → progress=1.0,eta=0。
        # 音频秒数当 total_steps 单位(spec callout 「合成 n/N秒 · ETA」就是这个单位)。
        total_units = max(1, int(round(result.duration_seconds)))
        pt.finish(
            total_units,
            detail=f"tts_synth done ({result.duration_seconds:.2f}s)",
        )

        return InferenceResult(
            media_type=f"audio/{result.format}",
            data=result.audio_bytes,
            metadata={
                "sample_rate": result.sample_rate,
                "duration_seconds": result.duration_seconds,
                "format": result.format,
            },
            usage=UsageMeter(
                audio_seconds=result.duration_seconds,
                latency_ms=latency_ms,
            ),
        )

    def unload(self) -> None:
        # round8:同 #234(image adapter)—— 只置 _model=None 不够,CUDA caching allocator
        # 仍持 GPU 块、监控显示占用、同卡再装可能 OOM。显式 gc + empty_cache 让显存真降。
        # 有额外 GPU 属性的引擎(moss _processor / voxcpm _voxcpm)override 先清自己的再 super。
        self._model = None
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — 清缓存 best-effort,不可因它崩卸载
            pass

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @property
    def supported_voices(self) -> list[str]:
        return ["default"]
