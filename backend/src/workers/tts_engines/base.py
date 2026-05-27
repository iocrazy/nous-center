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


def _make_tts_emit(progress_callback: Any) -> Any:
    """PR-1b:progress_callback 兼容层 —— 新契约 `(done, total, *, stage, progress,
    detail, step_latency_ms, eta_ms)`;旧 fake 可能只接 (done, total)。先发全量,
    TypeError 降级到 (done, total)。同 image_modular._make_emit 但 TTS 不带 preview_url。
    `progress_callback=None` → no-op,简化调用方。"""
    if progress_callback is None:
        def _noop(_done: int, _total: int, **_kw: Any) -> None:
            return None
        return _noop

    def _emit(done: int, total: int, **extras: Any) -> None:
        try:
            progress_callback(done, total, **extras)
            return
        except TypeError:
            pass
        progress_callback(done, total)

    return _emit


async def _tts_progress_ticker(
    emit: Any, t0: float, est_total_sec: float, est_total_units: int,
) -> None:
    """PR-1b:每 ~300ms 估 elapsed/est_total 发 progress 帧。progress 上限 0.95
    (给 end 帧 1.0 留增长空间,不显示「100% 然后下降」)。step 按 elapsed/sec 取整,
    eta 按 max(0, est_total_sec - elapsed) 折算 ms。
    被 cancel 时静默退出(infer 的 finally 负责 await task)。
    """
    import time

    try:
        while True:
            await asyncio.sleep(0.3)
            elapsed = time.monotonic() - t0
            progress = min(0.95, elapsed / est_total_sec)
            step = min(est_total_units, max(0, int(round(elapsed))))
            eta_ms = max(0, int((est_total_sec - elapsed) * 1000))
            emit(
                step, est_total_units,
                stage="tts_synth", progress=progress,
                detail=f"tts_synth {step}/{est_total_units}s",
                step_latency_ms=300, eta_ms=eta_ms,
            )
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

        # PR-1b:start 帧 —— 让 UI 在长合成里第一时间显示「正在合成」,不黑屏。
        # total 估计从文本长度推算(后面 ticker / end 会逐步收敛到真值)。
        t0 = time.monotonic()
        # 估秒数:经验值 ~0.08 秒/字符(中文密度;英文略快,折中保守)。
        # 实际值在 end 帧由 result.duration_seconds 回填,中间的 ticker 用这个估值算 ETA。
        est_total_sec = max(1.0, len(req.text) * 0.08)
        est_total_units = max(1, int(round(est_total_sec)))
        _tts_emit = _make_tts_emit(progress_callback)
        _tts_emit(
            0, est_total_units,
            stage="tts_synth", progress=0.0,
            detail="tts_synth start", step_latency_ms=None, eta_ms=None,
        )

        # spec §4.4 升级:boundary cancel(synthesize 调用前查一次;调用中通过
        # cancel_flag 透传给支持 step-level interrupt 的具体 engine)。这里仅
        # boundary 已经修正了「点 cancel 但 GPU 仍跑完整轮」的老问题。
        if cancel_flag is not None and cancel_flag.is_set():
            raise asyncio.CancelledError()

        # PR-1b:periodic ticker —— synthesize 在 to_thread 跑,event loop 同时跑
        # 一个 ticker 每 ~300ms 估算 elapsed/est_total 发 progress 帧。所有 TTS engine
        # 不需 per-engine streaming 也能有 L3 进度颗粒(前端「合成中 · ETA」立刻有反馈)。
        # 真值在 end 帧才回填;中间 progress 上限 0.95(给 end 帧留增长空间,不至于
        # 中间冲到 100% 后又回退 / 显示倒退)。
        ticker_task = None
        if progress_callback is not None:
            ticker_task = asyncio.create_task(
                _tts_progress_ticker(_tts_emit, t0, est_total_sec, est_total_units))

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

        # PR-1b:end 帧 —— 真实 duration_seconds 回填到 total(秒数,1-based 单位),
        # detail 给前端展示「✓ N.NN 秒」。step=total → progress=1.0,eta=0。
        # 把音频秒数当 total_steps 单位(整数化,frontend 显示「3/3 秒」更易读;
        # spec callout 文字模板「合成 n/N秒 · ETA」就是这个单位)。
        total_units = max(1, int(round(result.duration_seconds)))
        _tts_emit(
            total_units, total_units,
            stage="tts_synth", progress=1.0,
            detail=f"tts_synth done ({result.duration_seconds:.2f}s)",
            step_latency_ms=latency_ms, eta_ms=0,
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
        self._model = None

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @property
    def supported_voices(self) -> list[str]:
        return ["default"]
