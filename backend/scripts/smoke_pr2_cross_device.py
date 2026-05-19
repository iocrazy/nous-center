"""PR-2 manual smoke — cross-device Flux2-Klein-9B via ImageSampler.

Verifies spec §8 PR-2: image_generate runs with components on different GPUs
(transformer→cuda:1 Pro 6000, text_encoder→cuda:0 3090, vae→cuda:2 3090) and
produces a valid PNG.

Assumes hardware: cuda:0=3090, cuda:1=Pro 6000 96GB, cuda:2=3090 (PR #111
forced CUDA_DEVICE_ORDER=PCI_BUS_ID).

Run:
    cd backend && NOUS_DISABLE_RUNNER_SPAWN=1 .venv/bin/python scripts/smoke_pr2_cross_device.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from src.services.inference.base import ImageRequest
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
FLUX2_KLEIN_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B"


async def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.version.cuda} count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f}GB")

    components = {
        "unet": ComponentSpec(
            kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device="cuda:1", dtype="bfloat16",  # Pro 6000 (largest VRAM)
        ),
        "clip": ComponentSpec(
            kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device="cuda:0", dtype="bfloat16",  # 3090 #1
        ),
        "vae": ComponentSpec(
            kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device="cuda:2", dtype="bfloat16",  # 3090 #2
        ),
    }

    print("\nBuilding adapter via from_components")
    adapter = DiffusersImageBackend.from_components(components, pipeline_class="Flux2KleinPipeline")
    print("Loading cross-device")
    await adapter.load()
    print(f"  transformer.device={next(adapter._pipe.transformer.parameters()).device}")
    print(f"  text_encoder.device={next(adapter._pipe.text_encoder.parameters()).device}")
    print(f"  vae.device={next(adapter._pipe.vae.parameters()).device}")

    print("\nRunning 25-step inference at 512x512")
    t0 = time.monotonic()
    req = ImageRequest(
        request_id="smoke",
        prompt="a colorful hot air balloon over a green valley at sunset",
        seed=12345,
        steps=25,
        width=512, height=512,
    )
    result = await adapter.infer(req)
    elapsed = time.monotonic() - t0
    print(f"Inference: {elapsed:.1f}s, output size={len(result.data) // 1024}KB")

    # Save the image
    out_path = Path("/tmp/pr2_cross_device.png")
    out_path.write_bytes(result.data)
    print(f"Image saved to {out_path}")

    print("\nCross-device PR-2 smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
