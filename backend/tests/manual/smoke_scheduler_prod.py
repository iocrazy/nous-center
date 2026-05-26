"""SMOKE — PR-2 采样器/调度器经**生产路径**真模型验(needs GPU)。

验 `ModularImageBackend._apply_scheduler` 在真 diffusers 上换 pipe.scheduler 后出图正确 + 有差异:
同 prompt/seed/cfg,扫 (sampler_name, scheduler) 组合,对比 baseline(euler/normal)的 SSIM。
判据:每个组合都能出图(不崩,尤其 karras+use_dynamic_shifting 可能冲突),且 SSIM<1(scheduler 真生效)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_scheduler_prod.py
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

PROMPT = "a photo of a red fox in autumn leaves, sharp focus, detailed, golden hour"
SEED, STEPS, SIZE, CFG = 42, 25, 1024, 4.0

# 受支持组合(euler × 4 sigma 调度;真模型应全部出图)
COMBOS = [
    ("euler", "normal"),    # baseline
    ("euler", "karras"),
    ("euler", "exponential"),
    ("euler", "beta"),
]
# 不受支持(应被 _validate_sampler_scheduler 清晰拦截,不出图也不崩 diffusers)
UNSUPPORTED = [("heun", "normal"), ("lcm", "normal")]


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"[sched-prod] SSIM 跳过({e})")
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
    print(f"[sched-prod] {torch.cuda.get_device_properties(dev).name} device={DEVICE}")
    OUT.mkdir(exist_ok=True)
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")

    results: dict[str, Path] = {}
    for sampler, sched in COMBOS:
        tag = f"{sampler}_{sched}"
        try:
            res = await adapter.infer(ImageRequest(
                request_id=tag, prompt=PROMPT, steps=STEPS, width=SIZE, height=SIZE,
                seed=SEED, cfg_scale=CFG, sampler_name=sampler, scheduler=sched))
            p = OUT / f"sched_{tag}.png"
            p.write_bytes(res.data)
            results[tag] = p
            print(f"[sched-prod] {tag:18} OK → {p.name} ({res.usage.latency_ms}ms) "
                  f"sched_key={adapter._sched_key}")
        except Exception as e:  # noqa: BLE001
            print(f"[sched-prod] {tag:18} FAIL — {type(e).__name__}: {e}")

    base = results.get("euler_normal")
    if base:
        print("\n[sched-prod] === SSIM vs euler/normal(<1 = scheduler 真生效)===")
        for tag, p in results.items():
            if tag == "euler_normal":
                continue
            s = _ssim(base, p)
            print(f"[sched-prod]   SSIM(euler/normal, {tag:18}) = {s:.4f}" if s is not None else f"  {tag} n/a")

    # 不支持的采样器 → 必须清晰报错(不出图、不崩 diffusers)
    print("\n[sched-prod] === 不支持组合应清晰报错 ===")
    fail_loud_ok = True
    for sampler, sched in UNSUPPORTED:
        try:
            await adapter.infer(ImageRequest(
                request_id=f"{sampler}_{sched}", prompt=PROMPT, steps=STEPS, width=SIZE,
                height=SIZE, seed=SEED, cfg_scale=CFG, sampler_name=sampler, scheduler=sched))
            print(f"[sched-prod]   {sampler}/{sched}: 未报错(BAD —— 应被拦截)")
            fail_loud_ok = False
        except ValueError as e:
            print(f"[sched-prod]   {sampler}/{sched}: ✓ 清晰报错 — {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[sched-prod]   {sampler}/{sched}: 崩了非 ValueError(BAD)— {type(e).__name__}: {e}")
            fail_loud_ok = False

    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"\n[sched-prod] {len(results)}/{len(COMBOS)} 受支持组合出图,peak_vram={peak:.0f}MB")
    ok = len(results) == len(COMBOS) and fail_loud_ok
    print("[sched-prod] PASS" if ok else "[sched-prod] PARTIAL — 见上")


if __name__ == "__main__":
    asyncio.run(main())
