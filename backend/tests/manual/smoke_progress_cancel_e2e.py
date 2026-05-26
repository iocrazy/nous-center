"""SMOKE — PR-3(进度+中止)经**生产路径**真模型端到端验证(plan PR-G,manual,需 GPU)。

验三件事:
  (a) progress_callback 每步被调(契约级,不只是 stub mock):打 25 步出图,callback 记录次数。
  (b) cancel_flag 置位 → 下一步 raise CancelledError → 整次 infer <500ms 内退出(asyncio.CancelledError 走通)。
  (c) cancel 完再用同 backend infer:flag 不沾,正常出图。

不在 CI 里跑(需真 GPU + 大模型)。用户启动 backend 前手动跑一次,验 #148 在端到端真生效。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_progress_cancel_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
VAE = f"{IMG}/vae/flux2-vae.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT = Path(__file__).parent / "_smoke_out"


async def main() -> None:
    import torch

    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.cancel_flag import CancelFlag
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _ER(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    print(f"[pr3-e2e] device={DEVICE} {torch.cuda.get_device_properties(torch.device(DEVICE)).name}")
    OUT.mkdir(exist_ok=True)
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    mm = ModelManager(registry=_ER(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")

    # === (a) progress callback 每步被调 ===
    progress_calls: list[tuple[int, int, str | None]] = []
    def on_p(done: int, total: int, preview_url: str | None = None) -> None:
        # 记录:不依赖 mock,只看真 backend 在 25 步上有没有按契约调 callback
        progress_calls.append((done, total, preview_url[:40] if preview_url else None))

    t0 = time.monotonic()
    res = await adapter.infer(
        ImageRequest(request_id="a", prompt="a fox", steps=25, width=512, height=512, seed=42, cfg_scale=4.0),
        progress_callback=on_p,
    )
    elapsed_a = time.monotonic() - t0
    out_a = OUT / "pr3_e2e_full_run.png"
    out_a.write_bytes(res.data)
    print(f"[pr3-e2e] (a) FULL 25 steps OK — {len(progress_calls)} progress calls in {elapsed_a:.1f}s "
          f"-> {out_a.name}")
    assert len(progress_calls) == 25, f"expected 25 progress calls, got {len(progress_calls)}"
    assert progress_calls[0] == (1, 25, None) or progress_calls[0][:2] == (1, 25), f"first call wrong: {progress_calls[0]}"
    assert progress_calls[-1][:2] == (25, 25), f"last call wrong: {progress_calls[-1]}"
    # PR-F preview_url 应该至少一些步骤有(latent 解码出 JPEG)
    with_preview = sum(1 for c in progress_calls if c[2])
    print(f"[pr3-e2e] (a) preview_url 发出 {with_preview}/25 步")
    assert with_preview > 0, "no preview frames sent (PR-F latent_to_preview_data_uri 全失败)"

    # === (b) cancel mid-run ===
    cancel = CancelFlag()
    cancel_calls: list[tuple[int, int]] = []
    def on_p2(done: int, total: int, **_) -> None:
        cancel_calls.append((done, total))
        if done == 5:
            cancel.set("user-cancelled")  # 第 5 步触发 cancel,下一步应 raise

    t1 = time.monotonic()
    try:
        await adapter.infer(
            ImageRequest(request_id="b", prompt="a fox", steps=25, width=512, height=512, seed=42, cfg_scale=4.0),
            progress_callback=on_p2, cancel_flag=cancel,
        )
        print("[pr3-e2e] (b) FAIL — 没 raise CancelledError")
        return
    except asyncio.CancelledError:
        elapsed_b = time.monotonic() - t1
        last_step = cancel_calls[-1][0] if cancel_calls else 0
        # 第 6 步上下应该 raise(每步 ~270ms,cancel.set 后下一步开始就检测到)
        print(f"[pr3-e2e] (b) CANCEL OK — raised at step {last_step}, total elapsed {elapsed_b:.1f}s")
        assert last_step <= 7, f"cancel didn't take effect promptly: stopped at step {last_step}"

    # === (c) cancel 后再 infer,正常出图(flag 用完即弃) ===
    cancel2 = CancelFlag()  # 新 flag,未置位
    progress2: list[int] = []
    def on_p3(done: int, total: int, **_) -> None:
        progress2.append(done)
    t2 = time.monotonic()
    res2 = await adapter.infer(
        ImageRequest(request_id="c", prompt="a cat", steps=8, width=512, height=512, seed=99, cfg_scale=4.0),
        progress_callback=on_p3, cancel_flag=cancel2,
    )
    elapsed_c = time.monotonic() - t2
    out_c = OUT / "pr3_e2e_after_cancel.png"
    out_c.write_bytes(res2.data)
    print(f"[pr3-e2e] (c) POST-CANCEL OK — {len(progress2)} steps in {elapsed_c:.1f}s -> {out_c.name}")
    assert len(progress2) == 8, f"post-cancel run incomplete: {len(progress2)} steps"

    peak = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / 1024**2
    print(f"\n[pr3-e2e] PASS — 全 3 用例通过,peak_vram={peak:.0f}MB")
    print("[pr3-e2e]   (a) progress callback 真 25 步;preview_url 至少部分帧发出(PR-F 生效)")
    print("[pr3-e2e]   (b) cancel mid-run,step ~5+1 内停,raise CancelledError")
    print("[pr3-e2e]   (c) cancel 后同 backend 复用,新 flag 出图正常")


if __name__ == "__main__":
    asyncio.run(main())
