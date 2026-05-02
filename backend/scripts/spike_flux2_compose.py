"""PR-Spike: Verify Flux.2-Klein-9B-True-V2 composes in diffusers.

Goal: confirm the design doc's claim that we can load DiT (bf16) +
Qwen3 text encoder (fp8) + VAE separately and compose into a working
pipeline, before committing to PR-0 v2 ABC + 7 adapter migration.

Failure here means design doc P7 is wrong and we need to either:
  - upgrade diffusers / patch
  - write a custom pipeline subclass
  - fall back to ComfyUI subprocess (P2 design rejected this path)

Stages — run in order, each prints PASS/FAIL with diagnostic:
  1. import_smoke   — diffusers has the right classes (no GPU touched)
  2. config_check   — verify the file paths exist + report sizes
  3. dit_load       — load Flux2-Klein DiT bf16 (~17GB, GPU)
  4. encoder_load   — load Qwen3-8B fp8 (~8GB, GPU)
  5. vae_load       — load Flux2 VAE (~321MB, GPU)
  6. compose        — wire into FluxPipeline / Flux2Pipeline
  7. inference      — generate one 512x512 image with euler/25 steps

Usage:
    cd backend && uv run --extra image python scripts/spike_flux2_compose.py [stage]

If [stage] is omitted, runs stages 1-2 only (no GPU).
Pass `--full` to run all stages including ~120s GPU load.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Match design doc P7 paths
DIT_PATH = Path(
    "/media/heygo/Program/models/nous/image/diffusion_models/"
    "Flux2-Klein-9B-True-V2/Flux2-Klein-9B-True-v2-bf16.safetensors"
)
ENCODER_PATH = Path(
    "/media/heygo/Program/models/comfyui/models/text_encoders/"
    "qwen_3_8b_fp8mixed.safetensors"
)
VAE_PATH = Path(
    "/media/heygo/Program/models/nous/image/vae/flux2-vae.safetensors"
)


def stage(name: str):
    def deco(fn):
        def wrapper(*a, **kw):
            print(f"\n=== STAGE: {name} ===")
            t0 = time.time()
            try:
                result = fn(*a, **kw)
                dt = time.time() - t0
                print(f"PASS [{name}] {dt:.1f}s")
                return result
            except Exception as e:
                dt = time.time() - t0
                print(f"FAIL [{name}] {dt:.1f}s — {type(e).__name__}: {e}")
                raise

        return wrapper

    return deco


@stage("1. import_smoke")
def import_smoke() -> dict:
    """Verify diffusers has Flux.2 classes. No GPU."""
    import diffusers
    print(f"diffusers version: {diffusers.__version__}")

    found = {}

    # Try Flux.2-specific pipeline first (preferred)
    for cls_name in ["Flux2Pipeline", "FluxPipeline"]:
        try:
            cls = getattr(diffusers, cls_name)
            found[cls_name] = cls.__module__
            print(f"  {cls_name}: {cls.__module__}")
        except AttributeError:
            print(f"  {cls_name}: MISSING")
            found[cls_name] = None

    for cls_name in ["FluxTransformer2DModel", "AutoencoderKL"]:
        cls = getattr(diffusers, cls_name)
        found[cls_name] = cls.__module__
        print(f"  {cls_name}: {cls.__module__}")

    # Qwen3 encoder loader
    import transformers
    print(f"transformers version: {transformers.__version__}")
    for cls_name in ["AutoModel", "AutoTokenizer", "Qwen2Model", "Qwen3Model"]:
        try:
            cls = getattr(transformers, cls_name)
            found[cls_name] = cls.__module__
            print(f"  {cls_name}: OK")
        except AttributeError:
            print(f"  {cls_name}: MISSING")
            found[cls_name] = None

    if not found.get("Flux2Pipeline") and not found.get("FluxPipeline"):
        raise RuntimeError(
            "Neither Flux2Pipeline nor FluxPipeline found in diffusers. "
            "Need to upgrade diffusers main or write custom pipeline."
        )
    return found


@stage("2. config_check")
def config_check() -> dict:
    """Verify file paths exist + report sizes. No GPU."""
    paths = {"dit": DIT_PATH, "encoder": ENCODER_PATH, "vae": VAE_PATH}
    sizes = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{name}: {p} does not exist")
        size_gb = p.stat().st_size / (1024**3)
        sizes[name] = size_gb
        print(f"  {name}: {p} ({size_gb:.2f} GB)")
    total = sum(sizes.values())
    print(f"  TOTAL: {total:.2f} GB (single 24GB 3090 fits with cpu_offload)")
    return sizes


@stage("3. dit_load")
def dit_load():
    """Load Flux2-Klein DiT bf16. ~17GB, GPU."""
    import torch
    from diffusers import FluxTransformer2DModel

    print(f"  CUDA: {torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    transformer = FluxTransformer2DModel.from_single_file(
        str(DIT_PATH),
        torch_dtype=torch.bfloat16,
    )
    print(f"  loaded class: {type(transformer).__name__}")
    print(f"  config: {transformer.config}")
    return transformer


@stage("4. encoder_load")
def encoder_load():
    """Load Qwen3-8B fp8 as text encoder. ~8GB, GPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # NOTE: This is the unknown — Qwen3 fp8 single-file loading is the
    # exact compose gap outside voice flagged. We try several approaches.

    # Approach 1: AutoModel.from_pretrained pointing at parent dir (ComfyUI style
    # has the safetensors as a single file in models/text_encoders/, not a HF dir)
    # Try direct from_pretrained first — may need a config.json companion
    parent_dir = ENCODER_PATH.parent
    print(f"  trying AutoModel.from_pretrained({parent_dir})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(parent_dir))
        encoder = AutoModel.from_pretrained(
            str(parent_dir),
            torch_dtype=torch.bfloat16,  # fp8 weights, bf16 compute
            device_map="cuda:0",
        )
        print(f"  loaded class: {type(encoder).__name__}")
        return encoder, tokenizer
    except Exception as e:
        print(f"  AutoModel.from_pretrained failed: {e}")
        print("  This is the expected gap — Qwen3 fp8 needs custom loader path")
        print("  TODO: implement single-file Qwen3 fp8 loader in DiffusersImageBackend")
        raise


@stage("5. vae_load")
def vae_load():
    """Load Flux2 VAE. ~321MB, GPU."""
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_single_file(
        str(VAE_PATH),
        torch_dtype=torch.bfloat16,
    )
    print(f"  loaded class: {type(vae).__name__}")
    return vae


@stage("6. compose")
def compose(transformer, encoder, tokenizer, vae):
    """Compose into FluxPipeline / Flux2Pipeline."""
    import diffusers

    pipeline_cls = getattr(diffusers, "Flux2Pipeline", None) or diffusers.FluxPipeline
    print(f"  using: {pipeline_cls.__name__}")

    # Flux.1 expects text_encoder + text_encoder_2; Flux.2 may differ.
    # Try Flux.2 single-encoder shape first.
    try:
        pipe = pipeline_cls(
            transformer=transformer,
            text_encoder=encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=None,  # use default
        )
        print(f"  composed: {type(pipe).__name__}")
    except TypeError as e:
        print(f"  Flux.2 single-encoder shape failed: {e}")
        print("  May need text_encoder_2 / tokenizer_2 args (Flux.1 shape)")
        raise

    # Apply offload (production strategy)
    pipe.enable_model_cpu_offload()
    print("  enabled model_cpu_offload")
    return pipe


@stage("7. inference")
def inference(pipe):
    """Generate one 512x512 image (smaller than V0 1024 to save time)."""
    import torch

    out = pipe(
        prompt="a cat in space, oil painting",
        num_inference_steps=10,  # spike: fewer steps = faster
        width=512,
        height=512,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    image = out.images[0]
    print(f"  generated image: {image.size}, mode={image.mode}")
    out_path = Path("/tmp/spike_flux2_output.png")
    image.save(out_path)
    print(f"  saved: {out_path}")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run GPU stages too (~120s + may OOM on contested GPU).",
    )
    parser.add_argument("stage", nargs="?", default=None)
    args = parser.parse_args()

    print("PR-Spike: Flux.2-Klein-9B + diffusers compose verification")
    print("=" * 60)

    import_smoke()
    config_check()

    if not args.full:
        print("\n--- Stages 1-2 done. Pass --full for GPU stages 3-7 ---")
        return 0

    transformer = dit_load()
    encoder, tokenizer = encoder_load()
    vae = vae_load()
    pipe = compose(transformer, encoder, tokenizer, vae)
    inference(pipe)

    print("\n=== ALL STAGES PASSED ===")
    print("Design doc P7 confirmed. PR-0 v2 ABC baseline can begin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
