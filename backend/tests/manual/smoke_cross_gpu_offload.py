"""PR-D2 真模型验:跨卡 offload(stash=cuda:1 Pro 6000 / compute=cuda:0 3090)。

验证 `_enable_cross_gpu_offload`:
  - 装载在 stash 卡,forward 时挪 compute 卡,出图后挪回 stash
  - 出图正确(SSIM vs cpu-offload baseline > 0.95 ≈ 等价 — 同 compute device 才能公平比,
    RNG 是 device-specific 的)
  - peak VRAM on compute 卡 << 总模型尺寸(只装单组件)
  - peak VRAM on stash 卡 ≈ 全部组件总和

公平对比设计:
  - baseline: device=COMPUTE + offload=cpu(同 compute device,3090 装大模型,慢但正确)
  - cross-gpu: device=COMPUTE + offload=STASH(权重 stash 在 Pro 6000)
  两者 device 相同 → RNG 在同一 device 上初始化 latents → 出图应等价(SSIM ≈ 1.0)

用法:
    cd backend
    uv run python tests/manual/smoke_cross_gpu_offload.py

需要至少 2 张 GPU + Flux2 单文件权重在标准路径。
约 4-6 分钟在 Pro 6000 + 3090(baseline cpu-offload 比 单卡 baseline 慢)。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
VAE = f"{IMG}/vae/flux2-vae.safetensors"

# 默认:stash=cuda:1(Pro 6000,96GB)+ compute=cuda:0(3090,24GB)。
COMPUTE = os.environ.get("SMOKE_COMPUTE", "cuda:0")
STASH = os.environ.get("SMOKE_STASH", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
SEED = 42
STEPS = 20
SIZE = 1024


async def _run(label: str, device: str, offload: str) -> tuple[Path, dict]:
    import torch

    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    # 始终对 COMPUTE 和 STASH 两个固定卡 reset + measure,
    # 不管这个 run 是 baseline 还是 cross — 两个 case 数据可比。
    compute_dev = torch.device(COMPUTE)
    stash_dev = torch.device(STASH)
    torch.cuda.reset_peak_memory_stats(compute_dev)
    torch.cuda.reset_peak_memory_stats(stash_dev)

    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=device,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=device, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=device, dtype="bfloat16"),
    }

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(
        components, "Flux2KleinPipeline", offload=offload)

    res = await adapter.infer(ImageRequest(
        request_id=f"d2-{label}", prompt=PROMPT, cfg_scale=4.0, negative_prompt="",
        steps=STEPS, width=SIZE, height=SIZE, seed=SEED,
    ))

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"smoke_d2_{label}.png"
    out.write_bytes(res.data)
    stats = {
        "compute_peak_mib": torch.cuda.max_memory_allocated(compute_dev) / 1024**2,
        "stash_peak_mib": torch.cuda.max_memory_allocated(stash_dev) / 1024**2,
        "latency_ms": getattr(res.usage, "latency_ms", None),
    }
    return out, stats


async def main() -> None:
    import numpy as np
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    if torch.cuda.device_count() < 2:
        print(f"[d2] !! 需要 ≥2 张 GPU,实测 {torch.cuda.device_count()};跳过")
        return

    print(f"[d2] compute={COMPUTE} stash={STASH}")
    print(f"[d2] compute device = {torch.cuda.get_device_properties(torch.device(COMPUTE)).name}")
    print(f"[d2] stash   device = {torch.cuda.get_device_properties(torch.device(STASH)).name}")

    # baseline:device=COMPUTE + offload=cpu。同 device 才能跟 cross-gpu 公平比 SSIM(RNG
    # 是 device-specific 的 —— 不同设备 seed 42 出不同 noise,SSIM 必然低)。
    # 用 cpu offload 让 3090(24GB)能装下 34GB 大模型(慢 3-5× 但出图正确)。
    print(f"[d2] baseline: device={COMPUTE}, offload=cpu(同 compute device + CPU offload 装大模型)")
    baseline_path, baseline_stats = await _run("baseline", COMPUTE, "cpu")
    print(f"[d2]   peak: compute={baseline_stats['compute_peak_mib']:.0f}"
          f"  stash={baseline_stats['stash_peak_mib']:.0f} MiB"
          f"  latency = {baseline_stats['latency_ms']:.0f} ms")

    # 跨卡 offload:device=COMPUTE(3090,24GB),offload=STASH(Pro 6000)。
    print(f"[d2] cross-gpu: device={COMPUTE}, offload={STASH}")
    cross_path, cross_stats = await _run("cross", COMPUTE, STASH)
    print(f"[d2]   peak: compute={cross_stats['compute_peak_mib']:.0f}"
          f"  stash={cross_stats['stash_peak_mib']:.0f} MiB"
          f"  latency = {cross_stats['latency_ms']:.0f} ms")

    # SSIM 对比 — 期望 > 0.95(等价出图,挪挪不该改变 deterministic 输出)。
    def _load(p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert("RGB"))

    s = ssim(_load(baseline_path), _load(cross_path), channel_axis=2, data_range=255)
    print(f"[d2] SSIM(baseline, cross-gpu) = {s:.4f}")
    verdict = "PASS" if s > 0.95 else "FAIL"
    print(f"[d2] verdict = {verdict}")
    # compute 卡 peak 应小于单组件最大 18GB(transformer)+ 工作空间 ~20-22GB。
    print(f"[d2] compute peak {cross_stats['compute_peak_mib']:.0f} MiB"
          f" (期望 < 24GB,3090 不爆)")


if __name__ == "__main__":
    asyncio.run(main())
