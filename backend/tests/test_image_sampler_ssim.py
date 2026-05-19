"""SSIM regression — ImageSampler must produce nearly identical output to
diffusers Flux2KleinPipeline.__call__ when both run on the same single device.

Why this matters: if our self-written denoise loop diverges from Pipeline's,
the spec §8 PR-2 success criterion fails. SSIM > 0.99 confirms math parity.

Cost: ~60s on Pro 6000 (two 25-step Flux2 inferences back-to-back).
Run with: NOUS_SSIM_TEST=1 pytest tests/test_image_sampler_ssim.py -m slow -v
Skipped by default in normal test runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

MODELS_ROOT = Path("/media/heygo/Program/models/nous")
FLUX2_KLEIN_DIR = MODELS_ROOT / "image/diffusers/Flux2-klein-9B"

skip_reason = "real Flux2-Klein-9B model required + GPU; gated by NOUS_SSIM_TEST=1"
pytestmark = pytest.mark.skipif(
    os.environ.get("NOUS_SSIM_TEST") != "1" or not FLUX2_KLEIN_DIR.exists(),
    reason=skip_reason,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_image_sampler_matches_pipeline_ssim_single_device():
    """Same prompt + seed + steps + single device → SSIM > 0.99 between
    our ImageSampler.sample and the stock Pipeline.__call__ output.

    Runs on cuda:1 (Pro 6000) — assumes Task 0 / PR #111 confirmed cuda:1
    = Pro 6000 under CUDA_DEVICE_ORDER=PCI_BUS_ID.
    """
    import io

    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_diffusers import DiffusersImageBackend

    PROMPT = "a small grey kitten sitting on a wooden table, soft natural lighting"
    SEED = 4242
    STEPS = 25
    DEVICE = "cuda:1"  # Pro 6000

    # ===== Baseline: stock Pipeline.__call__ single-device =====
    baseline_pipe = Flux2KleinPipeline.from_pretrained(
        FLUX2_KLEIN_DIR, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    baseline_out = baseline_pipe(
        prompt=PROMPT,
        num_inference_steps=STEPS,
        height=512, width=512,
        generator=gen,
    )
    baseline_img: Image.Image = baseline_out.images[0]
    # Free baseline pipeline to make room for the sampler's pipeline
    del baseline_pipe
    torch.cuda.empty_cache()

    # ===== Our: ImageSampler via DiffusersImageBackend.from_components, same DEVICE =====
    components = {
        "unet": ComponentSpec(
            kind="unet", adapter_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "transformer/diffusion_pytorch_model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
        "clip": ComponentSpec(
            kind="clip", clip_arch="flux2",
            file=str(FLUX2_KLEIN_DIR / "text_encoder/model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
        "vae": ComponentSpec(
            kind="vae",
            file=str(FLUX2_KLEIN_DIR / "vae/diffusion_pytorch_model.safetensors"),
            device=DEVICE, dtype="bfloat16",
        ),
    }
    adapter = DiffusersImageBackend.from_components(
        components, pipeline_class="Flux2KleinPipeline"
    )
    await adapter.load()
    req = ImageRequest(
        request_id="ssim-test",
        prompt=PROMPT,
        seed=SEED,
        steps=STEPS,
        width=512, height=512,
    )
    sampler_result = await adapter.infer(req)
    sampler_img = Image.open(io.BytesIO(sampler_result.data))

    # ===== Compare =====
    assert baseline_img.size == sampler_img.size == (512, 512)

    a = np.array(baseline_img.convert("RGB"))
    b = np.array(sampler_img.convert("RGB"))
    score = ssim(a, b, channel_axis=2, data_range=255)
    print(f"SSIM = {score:.4f}")

    # Spec §8 PR-2 acceptance criterion
    assert score > 0.99, f"ImageSampler output diverges from Pipeline: SSIM={score:.4f}"
