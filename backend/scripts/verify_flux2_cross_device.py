"""PR-1 Task 0 risk gate — verify diffusers Flux2KleinPipeline accepts cross-device components.

Flux2-klein-9B on disk uses Flux2KleinPipeline (Qwen3 text encoder + AutoencoderKLFlux2),
NOT the FLUX.2-dev Flux2Pipeline (Mistral3 + AutoencoderKL). Pipeline class is determined
by model_index.json `_class_name`.

Runs manually before the rest of PR-1 work proceeds. Exit 0 = green light to continue.
Exit 1 = stop PR-1; activate spec §5.2 fallback (single-device assembly only).

Usage:
    cd backend && .venv/bin/python scripts/verify_flux2_cross_device.py
"""
from __future__ import annotations

import os

# Must run BEFORE any torch import — see PR #111. Without this torch uses
# FASTEST_FIRST ordering and cuda:0/cuda:1 indices flip vs nvidia-smi.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
from pathlib import Path

import torch

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
MODEL_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B"
TRANSFORMER_DIR = MODEL_DIR / "transformer"
TEXT_ENCODER_DIR = MODEL_DIR / "text_encoder"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
VAE_DIR = MODEL_DIR / "vae"
SCHEDULER_DIR = MODEL_DIR / "scheduler"

# Device targets for the test (current hardware: cuda:0=3090, cuda:1=Pro 6000, cuda:2=3090)
UNET_DEVICE = "cuda:1"   # largest VRAM, holds 18GB transformer
CLIP_DEVICE = "cuda:0"   # 3090 #1 holds 6GB text_encoder
VAE_DEVICE = "cuda:2"    # 3090 #2 holds 0.4GB vae


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f}GB")

    from diffusers import (
        AutoencoderKLFlux2,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading transformer from {TRANSFORMER_DIR} -> {UNET_DEVICE}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=torch.bfloat16
    ).to(UNET_DEVICE)
    print(f"  transformer.device={next(transformer.parameters()).device}")

    print(f"\nLoading text_encoder from {TEXT_ENCODER_DIR} -> {CLIP_DEVICE}")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        TEXT_ENCODER_DIR, torch_dtype=torch.bfloat16
    ).to(CLIP_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"  text_encoder.device={next(text_encoder.parameters()).device}")

    print(f"\nLoading vae from {VAE_DIR} -> {VAE_DEVICE}")
    vae = AutoencoderKLFlux2.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).to(VAE_DEVICE)
    print(f"  vae.device={next(vae.parameters()).device}")

    print(f"\nLoading scheduler from {SCHEDULER_DIR}")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SCHEDULER_DIR)

    print("\nAssembling Flux2KleinPipeline with cross-device components")
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )

    print("\nRunning 2-step inference as smoke test")
    out = pipe(prompt="a cat", num_inference_steps=2, height=512, width=512, generator=torch.Generator("cuda:1").manual_seed(0))
    img = out.images[0]
    print(f"  output image size={img.size}")
    if img.size != (512, 512):
        print(f"  ERROR expected (512,512) got {img.size}")
        return 1

    print("\n✅ Cross-device Flux2KleinPipeline assembly works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
