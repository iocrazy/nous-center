"""Z-Image-Turbo 独立冒烟(P1 de-risk,spec 2026-06-07)。standalone,需 GPU + 已下权重。

确认下载的 Z-Image-Turbo 权重 + 钉的 diffusers `ZImagePipeline` 真能端到端出图,
再据真实 API 写 ModularImageBackend 的 z-image 分支(避免按文档猜的 API 返工)。

Z-Image-Turbo:distilled → **guidance_scale=0**,8 步;text_encoder=Qwen3,vae=AutoencoderKL,
scheduler=FlowMatchEuler(见 model_index.json)。

用法(落 Pro 6000 cuda:1,~27G 权重 bf16 装得下):
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage.py
"""
from __future__ import annotations

import os

# standalone:import torch 前固定 PCI_BUS_ID(否则 cuda:1 可能是 3090 → OOM,见 CLAUDE.md)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
from pathlib import Path

REPO = os.environ.get(
    "SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
PROMPT = os.environ.get(
    "SMOKE_PROMPT", "a photo of a red fox sitting in autumn leaves, sharp focus, detailed")
STEPS = int(os.environ.get("SMOKE_STEPS", "8"))
SIZE = int(os.environ.get("SMOKE_SIZE", "1024"))
OUT = Path(__file__).parent / "_smoke_out" / "smoke_zimage.png"


def _engine_smoke() -> int:
    """走 ModularImageBackend 引擎路径(P1a 真实代码:_build_zimage_pipe + infer guidance=0),
    不只是裸 ZImagePipeline。SMOKE_VIA_ENGINE=1 启用。"""
    import asyncio
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    async def _run() -> int:
        be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16",
                                 pipeline_class="ZImagePipeline")
        await be.load(DEVICE)
        t = time.monotonic()
        res = await be.infer(ImageRequest(
            request_id="zimg-engine", prompt=PROMPT, cfg_scale=4.0,  # cfg 应被引擎强制为 0
            negative_prompt="", steps=STEPS, width=SIZE, height=SIZE, seed=42))
        OUT.parent.mkdir(exist_ok=True)
        out = OUT.with_name("smoke_zimage_engine.png")
        out.write_bytes(res.data)
        print(f"PASS(引擎路径)— ModularImageBackend(ZImagePipeline) 出图 → {out} "
              f"({int((time.monotonic()-t)*1000)}ms)")
        return 0
    return asyncio.run(_run())


def main() -> int:
    if os.environ.get("SMOKE_VIA_ENGINE") == "1":
        return _engine_smoke()
    import torch
    from diffusers import ZImagePipeline

    if not Path(REPO).exists():
        raise SystemExit(f"Z-Image 权重不存在: {REPO}(还没下完?)")

    print(f"加载 ZImagePipeline.from_pretrained({REPO}) → {DEVICE}")
    t0 = time.monotonic()
    pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to(DEVICE)
    load_s = time.monotonic() - t0
    print(f"  load {load_s:.1f}s")

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    t1 = time.monotonic()
    # distilled:guidance_scale=0(非零会掉质量,见 HF README)。
    out = pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=0.0,
               width=SIZE, height=SIZE, generator=gen)
    infer_s = time.monotonic() - t1
    img = out.images[0]
    OUT.parent.mkdir(exist_ok=True)
    img.save(OUT)
    print(f"  infer {infer_s:.1f}s ({STEPS} 步) → {OUT}")
    print(f"PASS — Z-Image 出图 {img.size}(load {load_s:.1f}s / infer {infer_s:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
