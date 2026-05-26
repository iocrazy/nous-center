"""SMOKE — 经**生产路径**验 true-CFG 修复(needs GPU)。

验 PR「true-cfg 修复」的产品代码:get_or_load_image_adapter → ModularImageBackend._build_klein_pipe
(标准 Flux2KleinPipeline, is_distilled=False)→ infer(cfg/negative 透传)。对照 spike_true_cfg(手搭 pipe),
本 smoke 走真实引擎入口,证明**我的接线**真出 true-CFG。

判据:同 seed,cfg=1.0(无 CFG)vs cfg=4.5+negative(true-CFG)→ SSIM 明显 < 1(< 0.9)→ 修复在生产路径生效。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_true_cfg_prod.py
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
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT = Path(__file__).parent / "_smoke_out"

PROMPT = ("a photo of a red fox sitting in autumn leaves, sharp focus, highly detailed, "
          "professional photography, golden hour lighting")
NEG = "blurry, low quality, distorted, deformed, washed out, jpeg artifacts"
SEED, STEPS, SIZE = 42, 25, 1024


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"[true-cfg-prod] SSIM 跳过(缺 skimage/PIL: {e})")
        return None
    a = np.asarray(Image.open(p1).convert("L"))
    b = np.asarray(Image.open(p2).convert("L"))
    return float(ssim(a, b))


async def main() -> None:
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

    dev = torch.device(DEVICE)
    print(f"[true-cfg-prod] {torch.cuda.get_device_properties(dev).name} device={DEVICE}")
    OUT.mkdir(exist_ok=True)
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    print(f"[true-cfg-prod] adapter={type(adapter).__name__} pipe={type(adapter._ensure_pipe()).__name__} "
          f"engine_class={adapter.pipeline_class}")

    async def run(tag: str, cfg: float, neg: str) -> Path:
        res = await adapter.infer(ImageRequest(
            request_id=tag, prompt=PROMPT, negative_prompt=neg,
            steps=STEPS, width=SIZE, height=SIZE, seed=SEED, cfg_scale=cfg))
        p = OUT / f"prod_{tag}.png"
        p.write_bytes(res.data)
        print(f"[true-cfg-prod] {tag:12} cfg={cfg} neg={'y' if neg else 'n'} "
              f"→ {p.name} engine={res.metadata['engine']} {res.usage.latency_ms}ms")
        return p

    a = await run("cfg1", 1.0, "")
    b = await run("cfg4_5_neg", 4.5, NEG)
    s = _ssim(a, b)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"\n[true-cfg-prod] SSIM(cfg1, cfg4.5+neg) = {s:.4f}" if s is not None else "[true-cfg-prod] SSIM n/a")
    print(f"[true-cfg-prod] peak_vram={peak:.0f}MB")
    if s is not None and s < 0.9:
        print("[true-cfg-prod] PASS — 生产路径 true-CFG 生效(SSIM<0.9,cfg/negative 真改变出图)")
    elif s is not None:
        print(f"[true-cfg-prod] FAIL — SSIM={s:.4f} 太高,cfg 可能仍未生效(查管线/is_distilled)")


if __name__ == "__main__":
    asyncio.run(main())
