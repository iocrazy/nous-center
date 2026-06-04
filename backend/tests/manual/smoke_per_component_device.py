"""逐组件选卡真模型验(spec 2026-06-04 / PR-A)。

验 `ModularImageBackend` 逐组件跨卡放置:transformer/clip/vae 各落不同卡,跨卡张量流由
forward 边界 ModelHook 透明搬运,出图与「整模型单卡」基线等价。

公平对比:两个 case 的 **transformer(compute 锚点)都在 cuda:1** → _execution_device 一致
→ latents/noise 同 device 上的 RNG 一致 → 出图应等价。
  - baseline:三组件全 cuda:1(整模型单卡,旧路径)
  - cross   :transformer cuda:1 / clip <CLIP_DEV> / vae <VAE_DEV>(逐组件跨卡,新路径)
若逐组件放置 / 跨卡搬运有 bug → cross 崩或出错图(SSIM 掉)。

**默认只把 vae 移到另一张卡(clip 留 cuda:1)→ 同硬件 → SSIM ≈ 1.0(干净的放置正确性闸门)。**
注意:把 clip 移到**不同架构**的卡(cuda:0/cuda:2 是 3090 Ampere,cuda:1 是 Pro6000
Blackwell)会让 SSIM 掉到 ~0.95 —— 那是不同 GPU 的 FP 内核数值差异(硬件,非 bug),
出图构图一致。要看这效果:`SMOKE_CLIP_DEV=cuda:0 ...`(真机实测 full cross=0.9517,
vae-only=0.9995)。

用法:
    cd backend
    uv run python tests/manual/smoke_per_component_device.py                 # 默认 vae→cuda:2
    SMOKE_CLIP_DEV=cuda:0 uv run python tests/manual/smoke_per_component_device.py  # 含跨架构 clip

需要 ≥2 张 GPU + Flux2 单文件权重在标准路径。约 1-2 分钟(两次 infer + 一次跨卡 build)。
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

COMPUTE = os.environ.get("SMOKE_COMPUTE", "cuda:1")    # transformer / 锚点(Pro6000)
# 默认 clip 留 compute 卡(同硬件,干净闸门),只 vae 跨卡;SMOKE_CLIP_DEV=cuda:0 看跨架构漂移。
CLIP_DEV = os.environ.get("SMOKE_CLIP_DEV", "cuda:1")
VAE_DEV = os.environ.get("SMOKE_VAE_DEV", "cuda:2")    # 3090
OUT_DIR = Path(__file__).parent / "_smoke_out"
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
SEED, STEPS, SIZE = 42, 20, 1024


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"  (skip SSIM — {e})")
        return None
    a = np.asarray(Image.open(p1).convert("RGB"))
    b = np.asarray(Image.open(p2).convert("RGB"))
    return float(ssim(a, b, channel_axis=2))


async def _run(label: str, clip_dev: str, vae_dev: str) -> Path:
    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=COMPUTE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=clip_dev, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=vae_dev, dtype="bfloat16"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    res = await adapter.infer(ImageRequest(
        request_id=f"pcd-{label}", prompt=PROMPT, cfg_scale=4.0, negative_prompt="",
        steps=STEPS, width=SIZE, height=SIZE, seed=SEED,
    ))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"smoke_pcd_{label}.png"
    out.write_bytes(res.data)
    print(f"  [{label}] clip={clip_dev} vae={vae_dev} → {out.name} "
          f"({getattr(res.usage, 'latency_ms', None)}ms)")
    # 彻底释放,下个 case 重新装(cuda:1 上有 ~46GB 常驻,预算紧)。
    import gc

    import torch
    try:
        adapter.unload()
    except Exception:  # noqa: BLE001
        pass
    del adapter, mm, components
    gc.collect()
    for d in ("cuda:0", "cuda:1", "cuda:2"):
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device(d))
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    return out


async def main() -> int:
    print(f"逐组件选卡 smoke:transformer={COMPUTE} (锚点);"
          f"baseline=全 {COMPUTE} / cross=clip {CLIP_DEV}+vae {VAE_DEV}")
    # cross 先跑(cuda:1 最空时装 transformer);再跑 baseline。
    cross = await _run("cross", CLIP_DEV, VAE_DEV)
    base = await _run("baseline", COMPUTE, COMPUTE)
    s = _ssim(base, cross)
    if s is None:
        print("PASS(出图无异常;装 scikit-image 看 SSIM)")
        return 0
    print(f"SSIM(baseline, cross) = {s:.4f}")
    ok = s >= 0.97
    print("PASS — 逐组件跨卡出图与整模型单卡等价" if ok
          else "FAIL — 跨卡放置出图偏差过大(SSIM < 0.97)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
