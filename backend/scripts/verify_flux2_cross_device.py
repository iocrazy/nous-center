"""PR-1 Task 0 risk gate — verify diffusers Flux2Pipeline accepts cross-device components.

Runs manually before the rest of PR-1 work proceeds. Exit 0 = green light to continue.
Exit 1 = stop PR-1; activate spec §5.2 fallback (single-device assembly only).

Usage:
    cd backend && .venv/bin/python scripts/verify_flux2_cross_device.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
TRANSFORMER_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/transformer"
TEXT_ENCODER_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/text_encoder"
VAE_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B/vae"

# Device targets for the test (current hardware: cuda:0=3090, cuda:1=Pro 6000, cuda:2=3090)
UNET_DEVICE = "cuda:1"   # largest VRAM, holds 18GB transformer
CLIP_DEVICE = "cuda:0"   # 3090 #1 holds 6GB text_encoder
VAE_DEVICE = "cuda:2"    # 3090 #2 holds 0.4GB vae


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f}GB")

    from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
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
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_DIR)
    print(f"  text_encoder.device={next(text_encoder.parameters()).device}")

    print(f"\nLoading vae from {VAE_DIR} -> {VAE_DEVICE}")
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).to(VAE_DEVICE)
    print(f"  vae.device={next(vae.parameters()).device}")

    print("\nAssembling Flux2Pipeline with cross-device components")
    pipe = Flux2Pipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=None,  # let Pipeline pick default
    )

    print("\nRunning 2-step inference as smoke test")
    out = pipe(prompt="a cat", num_inference_steps=2, height=512, width=512, generator=torch.Generator("cuda:1").manual_seed(0))
    img = out.images[0]
    print(f"  output image size={img.size}")
    if img.size != (512, 512):
        print(f"  ERROR expected (512,512) got {img.size}")
        return 1

    print("\n✅ Cross-device Flux2Pipeline assembly works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
