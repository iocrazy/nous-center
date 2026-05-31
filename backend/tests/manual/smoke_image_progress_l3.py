"""PR-1a smoke — 真模型验 L3 progress event 时序(stage / step / latency / ETA)。

为什么必须 standalone:conftest mock torch + 无 GPU,引擎正确性只能这条路验([[feedback-verify-real-model]])。
本 smoke 直接 instantiate ModularImageBackend(走 image_modular.py 真 callback_on_step_end)
跑 Flux2Klein,捕获 progress_callback 每一帧,断言:

1. **stage 时序**:首个 event = text_encode(pipe() 前)→ N 个 dit_denoise(逐步)→ 末尾
   = vae_decode(pipe() 后)。无乱序、无漏发、无重复。
2. **step 单调递增**:dit_denoise step = 1, 2, ..., N(1-based,对齐 ComfyUI ProgressBar.update_absolute)。
3. **ETA 递减**:dit_denoise.eta_ms 末步 = 0(total - step = 0)且整体趋势递减(允许早期波动)。
4. **step_latency_ms 合理**:int >= 0;1024px Flux2 单步在 Pro 6000 大约 ~200-1500ms,在范围内。

用法(落 cuda:1 Pro 6000):
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_image_progress_l3.py

可选环境变量:
    SMOKE_MODEL  整模型路径(默认 Flux2-klein-9B HF-layout)
    SMOKE_STEPS  采样步数(默认 8,够看到 ETA 收敛)
    SMOKE_SIZE   图尺寸(默认 512,smoke 不求质量)
    SMOKE_DTYPE  权重精度(默认 bfloat16)
"""
from __future__ import annotations


import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # standalone:cuda:1=Pro 6000,对齐 nvidia-smi(torch 默认 FASTEST_FIRST 会翻卡)
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ROOT = Path(os.environ.get(
    "SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"))
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
DTYPE = os.environ.get("SMOKE_DTYPE", "bfloat16")
STEPS = int(os.environ.get("SMOKE_STEPS", "8"))
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
PROMPT = os.environ.get("SMOKE_PROMPT", "a serene mountain lake at dawn")


def _fmt(events: list[dict]) -> str:
    lines = []
    for i, e in enumerate(events):
        stage = e.get("stage", "?")
        step = e.get("step")
        total = e.get("total")
        lat = e.get("step_latency_ms")
        eta = e.get("eta_ms")
        lines.append(
            f"  [{i:2d}] stage={stage:<12} step={step}/{total} lat={lat}ms eta={eta}ms"
        )
    return "\n".join(lines)


async def main() -> None:
    if not MODEL_ROOT.exists():
        raise SystemExit(f"模型路径不存在:{MODEL_ROOT}(SMOKE_MODEL 覆盖)")

    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    events: list[dict] = []

    def on_progress(step: int, total: int, **extras) -> None:
        events.append({"step": step, "total": total, **extras})

    print(f"[smoke] device={DEVICE} dtype={DTYPE} steps={STEPS} size={SIZE}")
    t0 = time.monotonic()
    backend = ModularImageBackend(
        repo=str(MODEL_ROOT),
        device=DEVICE,
        dtype=DTYPE,
        pipeline_class="Flux2KleinPipeline",
        offload="none",
    )
    print(f"[smoke] adapter inited in {time.monotonic()-t0:.1f}s,跑 infer ...")

    t1 = time.monotonic()
    req = ImageRequest(
        request_id="smoke-l3", prompt=PROMPT, negative_prompt="",
        width=SIZE, height=SIZE, steps=STEPS, cfg_scale=4.0, seed=42,
    )
    res = await backend.infer(req, progress_callback=on_progress)
    infer_s = time.monotonic() - t1
    print(f"[smoke] infer 完成({infer_s:.1f}s,出图 {len(res.data)/1024:.0f}KB)")
    print(f"[smoke] events ({len(events)}):")
    print(_fmt(events))

    # —— 断言:stage 时序 ——
    stages = [e.get("stage") for e in events]
    assert stages[0] == "text_encode", f"首个 event 必须是 text_encode,实际:{stages[0]}"
    assert stages[-1] == "vae_decode", f"末尾 event 必须是 vae_decode,实际:{stages[-1]}"
    dit_indices = [i for i, s in enumerate(stages) if s == "dit_denoise"]
    assert len(dit_indices) == STEPS, f"dit_denoise event 数应 = STEPS({STEPS}),实际:{len(dit_indices)}"
    # 全连续:text_encode 后到 vae_decode 前全是 dit_denoise(无乱序)
    middle = stages[1:-1]
    assert all(s == "dit_denoise" for s in middle), f"中段 stage 必须全是 dit_denoise,实际:{middle}"

    # —— 断言:step 单调递增 1..N ——
    dit_events = [e for e in events if e.get("stage") == "dit_denoise"]
    steps_seq = [e["step"] for e in dit_events]
    assert steps_seq == list(range(1, STEPS + 1)), f"step 序列必须 1..{STEPS},实际:{steps_seq}"

    # —— 断言:step_latency_ms 合理(int >= 0)——
    for e in dit_events:
        lat = e.get("step_latency_ms")
        assert isinstance(lat, int) and lat >= 0, f"step_latency_ms 应为非负 int,实际:{lat!r}"

    # —— 断言:末步 ETA = 0(total - step = 0)——
    assert dit_events[-1]["eta_ms"] == 0, f"末步 eta_ms 应为 0,实际:{dit_events[-1]['eta_ms']}"

    # —— 断言:ETA 整体递减(末步 < 首步;不强单调,因首步包含 cuda 启动 / cudnn benchmark 抖动)——
    if STEPS >= 3:
        assert dit_events[-1]["eta_ms"] <= dit_events[0]["eta_ms"], (
            f"ETA 应递减,首步 {dit_events[0]['eta_ms']} → 末步 {dit_events[-1]['eta_ms']}")

    print("[smoke] ✓ stage 时序 / step 单调 / ETA 递减 全通过")


if __name__ == "__main__":
    asyncio.run(main())
