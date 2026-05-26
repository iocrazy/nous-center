"""Verify cfg/negative 真生效经生产路径 —— PR #144 (true-CFG) 是否真起作用。

跑 4 张图(同 seed/同 prompt,变 cfg/negative),算 SSIM 验证 cfg=1 跟 cfg>1
产出明显不同。预期:
  - SSIM(cfg1, cfg3.5)  < 0.95  — 蒸馏管线会 ≈1.0(cfg 被掐)
  - SSIM(cfg1, cfg5)    < 0.95
  - SSIM(cfg1, cfg4+neg)< 0.95

为什么这个 smoke 存在(不被 CI 跑也要保留):
  - 蒸馏 vs 非蒸馏 pipeline 的差异只在真模型出图时显形;CI 用 mock 跑过
    不代表 cfg/negative 真生效。memory `feedback_verify_real_model`。
  - 历史教训:PR #144 之前,nous 单文件借 Flux2-klein-9B(`is_distilled=true`)
    的 config 走蒸馏 block(`guidance=None`,无 negative 分支),cfg/negative
    全被掐 — CI 全绿但用户感知出图质量差。

历史 baseline(2026-05-26 实测 / NVIDIA RTX PRO 6000 / Flux2-Klein-9B-True-v2-bf16):
    cfg1.0 vs cfg3.5      = 0.6553
    cfg1.0 vs cfg5.0      = 0.6143
    cfg1.0 vs cfg4.0+neg  = 0.6112
    peak VRAM = 38.6 GB
    cfg=1: 3.79 it/s  /  cfg>1: 1.83 it/s(double-forward,预期 2× 慢)

跑这个之前先 git pull origin master。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_negative_cfg_prod.py

需 GPU。跑约 1-2 分钟(4 张 1024² × 20 步在 Pro 6000)。
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
OUT_DIR = Path(__file__).parent / "_smoke_out"

PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
NEG = "blurry, low quality, ugly, deformed, watermark, text"
SEED = 42
STEPS = 20
SIZE = 1024


async def main() -> None:
    import numpy as np
    import torch
    from PIL import Image

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
    print(f"[neg-cfg] {torch.cuda.get_device_properties(dev).name}")
    torch.cuda.reset_peak_memory_stats(dev)

    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")

    OUT_DIR.mkdir(exist_ok=True)
    cases = [
        ("cfg1.0",       1.0,  ""),
        ("cfg3.5",       3.5,  ""),
        ("cfg5.0",       5.0,  ""),
        ("cfg4.0+neg",   4.0,  NEG),
    ]
    paths: dict[str, Path] = {}
    for label, cfg, neg in cases:
        print(f"[neg-cfg] {label} cfg={cfg} neg={'Y' if neg else '-'}")
        res = await adapter.infer(ImageRequest(
            request_id=f"neg-cfg-{label}",
            prompt=PROMPT,
            negative_prompt=neg,
            cfg_scale=cfg,
            steps=STEPS,
            width=SIZE,
            height=SIZE,
            seed=SEED,
        ))
        out = OUT_DIR / f"smoke_negcfg_{label}.png"
        out.write_bytes(res.data)
        paths[label] = out

    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"[neg-cfg] peak VRAM: {peak:.1f} MiB")

    # —— SSIM 对照 —— cfg=1 是 baseline。
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("[neg-cfg] !! skimage 未装,跳过 SSIM(uv add scikit-image 后再跑)")
        return

    def _load(p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert("RGB"))

    baseline = _load(paths["cfg1.0"])
    print("[neg-cfg] SSIM(cfg1.0, x) — 期望 << 1.0(蒸馏管线会 ≈1.0)")
    verdict = "PASS"
    for label in ("cfg3.5", "cfg5.0", "cfg4.0+neg"):
        s = ssim(baseline, _load(paths[label]), channel_axis=2, data_range=255)
        mark = "✓" if s < 0.95 else "✗ (cfg 似乎没生效)"
        if s >= 0.95:
            verdict = "FAIL"
        print(f"[neg-cfg]   cfg1.0 vs {label:14s} = {s:.4f}  {mark}")
    print(f"[neg-cfg] verdict = {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
