"""PR-1b smoke — 真 TTS 引擎验 L3 progress event 时序(stage / step / latency / ETA)。

为什么必须 standalone:conftest mock torch + 无 GPU,引擎正确性只能这条路验
([[feedback-verify-real-model]])。本 smoke 加载真 TTS engine(默认 cosyvoice2),
跑一句较长文本(让 periodic ticker 有机会 fire >= 1 次),捕获 progress 事件:

1. **stage 全是 tts_synth**:start/ticker/end 三类事件 stage 一致。
2. **首尾帧**:第一帧 progress=0.0(start);末帧 progress=1.0 + eta_ms=0(end)。
3. **ticker 帧存在**(若文本足够长):中间至少一帧 progress 介于 0 ~ 0.95,
   detail 含 "tts_synth N/Ms" 格式。短文本可能瞬完,ticker 不 fire 也接受。
4. **end 帧 detail 含 duration_seconds**(回填真值)。

用法(落 cuda:1 / 长文本触发 ticker):
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_tts_progress.py

可选环境变量:
    SMOKE_ENGINE   引擎名(默认 cosyvoice2)
    SMOKE_MODEL    模型路径(默认按 engine 推断)
    SMOKE_TEXT     合成文本(默认一段长中文,确保 ticker 触发)
    SMOKE_DEVICE   GPU 设备(默认 cuda:1 = Pro 6000)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENGINE = os.environ.get("SMOKE_ENGINE", "cosyvoice2")
DEFAULT_MODEL_PATHS = {
    "cosyvoice2": "/media/heygo/Program/models/nous/speech/cosyvoice2-0.5b",
    "qwen3_tts_base": "/media/heygo/Program/models/nous/speech/qwen3-tts-1.7b-base",
    "indextts2": "/media/heygo/Program/models/nous/speech/indextts-2",
    "voxcpm2": "/media/heygo/Program/models/nous/speech/VoxCPM2",
}
MODEL_PATH = Path(os.environ.get(
    "SMOKE_MODEL", DEFAULT_MODEL_PATHS.get(ENGINE, "")))
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
TEXT = os.environ.get(
    "SMOKE_TEXT",
    "这是一段较长的测试文本,用来验证 TTS 引擎在合成过程中能够稳定发出 L3 颗粒度的进度事件,"
    "包括起始帧、周期性的 ticker 帧、以及结束帧。"
)


def _fmt(events: list[dict]) -> str:
    lines = []
    for i, e in enumerate(events):
        stage = e.get("stage", "?")
        step = e.get("step")
        total = e.get("total")
        prog = e.get("progress")
        lat = e.get("step_latency_ms")
        eta = e.get("eta_ms")
        det = e.get("detail", "")
        lines.append(
            f"  [{i:2d}] stage={stage:<10} step={step}/{total} prog={prog} "
            f"lat={lat} eta={eta} | {det}"
        )
    return "\n".join(lines)


def _resolve_engine_cls(engine_name: str):
    # Lazy import 各 engine module,触发 register_engine 装饰器。
    if engine_name == "cosyvoice2":
        from src.workers.tts_engines.cosyvoice2 import CosyVoice2Engine
        return CosyVoice2Engine
    if engine_name.startswith("qwen3_tts"):
        from src.workers.tts_engines.qwen3_tts import Qwen3TTSBaseEngine
        return Qwen3TTSBaseEngine
    if engine_name == "indextts2":
        from src.workers.tts_engines.indextts2 import IndexTTS2Engine
        return IndexTTS2Engine
    if engine_name == "voxcpm2":
        from src.workers.tts_engines.voxcpm2 import VoxCPM2Engine
        return VoxCPM2Engine
    raise SystemExit(f"未知 engine: {engine_name}(支持:cosyvoice2/qwen3_tts_base/indextts2/voxcpm2)")


async def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"模型路径不存在:{MODEL_PATH}(SMOKE_MODEL 覆盖,或先下载模型)")

    from src.services.inference.base import AudioRequest

    cls = _resolve_engine_cls(ENGINE)
    print(f"[smoke] engine={ENGINE} device={DEVICE} model={MODEL_PATH}")

    events: list[dict] = []

    def on_progress(step: int, total: int, **extras) -> None:
        events.append({"step": step, "total": total, **extras})

    t0 = time.monotonic()
    adapter = cls(paths={"main": str(MODEL_PATH)}, device=DEVICE)
    await adapter.load(DEVICE)
    print(f"[smoke] adapter loaded in {time.monotonic()-t0:.1f}s,跑 infer ...")

    t1 = time.monotonic()
    req = AudioRequest(request_id="smoke-l3-tts", text=TEXT, sample_rate=24000)
    res = await adapter.infer(req, progress_callback=on_progress)
    infer_s = time.monotonic() - t1
    print(f"[smoke] infer 完成({infer_s:.2f}s,音频 {len(res.data)/1024:.0f}KB,"
          f"duration={res.metadata.get('duration_seconds')}s)")
    print(f"[smoke] events ({len(events)}):")
    print(_fmt(events))

    # —— 断言 1:首末帧 stage = tts_synth(其它中间帧也必须是) ——
    stages = [e.get("stage") for e in events]
    assert all(s == "tts_synth" for s in stages), f"stage 必须全是 tts_synth,实际:{set(stages)}"

    # —— 断言 2:首帧 progress = 0.0(start) ——
    assert events[0]["progress"] == 0.0, f"首帧 progress 必须 = 0.0,实际:{events[0]['progress']}"

    # —— 断言 3:末帧 progress = 1.0 + eta = 0(end) ——
    end = events[-1]
    assert end["progress"] == 1.0, f"末帧 progress 必须 = 1.0,实际:{end['progress']}"
    assert end["eta_ms"] == 0, f"末帧 eta_ms 必须 = 0,实际:{end['eta_ms']}"
    assert "done" in (end.get("detail") or ""), f"末帧 detail 应含 'done',实际:{end.get('detail')!r}"

    # —— 断言 4:若 infer 耗时 > 0.5s,ticker 应至少 fire 一次(中间帧 progress 介于 0~0.95) ——
    if infer_s > 0.5:
        middle = events[1:-1]
        assert middle, f"infer 耗 {infer_s:.2f}s 但无 ticker 中间帧 —— ticker 没工作?"
        for e in middle:
            assert 0 < e["progress"] < 1.0, f"中间帧 progress 应 ∈(0,1),实际:{e['progress']}"

    print("[smoke] ✓ stage 时序 / start-end 帧 / ticker 触发 全通过")


if __name__ == "__main__":
    asyncio.run(main())
