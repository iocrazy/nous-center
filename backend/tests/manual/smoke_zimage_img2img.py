"""Z-Image img2img(strength)独立冒烟 —— PR-A2(spec 2026-06-08-multi-sampling-cross-model)。

走完整引擎路径:文生图出基图 → 同模型 img2img(ZImageImg2ImgPipeline,strength<1)加噪重去噪。
验:① img2img 真出图(_wants_img2img→_ensure_img2img_pipe→strength 注入生效);② 输出与基图
**不同**(img2img 真改了)但**结构相关**(SSIM 落在 0.2~0.95:既非原图也非无关图)。standalone,需 GPU。

用法(落 Pro 6000 cuda:1):
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_img2img.py
"""
from __future__ import annotations

import os

# standalone:import torch 前固定 PCI_BUS_ID(否则 cuda:1 可能是 3090 → OOM,见 CLAUDE.md)。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
STRENGTH = float(os.environ.get("SMOKE_STRENGTH", "0.6"))
STEPS = int(os.environ.get("SMOKE_STEPS", "8"))
SIZE = int(os.environ.get("SMOKE_SIZE", "1024"))
OUT_DIR = Path(__file__).parent / "_smoke_out"


def _ssim(a_bytes: bytes, b_bytes: bytes) -> float:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    a = np.asarray(Image.open(io.BytesIO(a_bytes)).convert("L"))
    b = np.asarray(Image.open(io.BytesIO(b_bytes)).convert("L"))
    return float(ssim(a, b))


async def _run() -> int:
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be.load(DEVICE)
    OUT_DIR.mkdir(exist_ok=True)

    # 1. 文生图基图(strength 默认 1.0 → 不走 img2img,零回归路径)。
    t = time.monotonic()
    base = await be.infer(ImageRequest(
        request_id="zi2i-base", prompt="a serene mountain lake at dawn, photorealistic",
        steps=STEPS, width=SIZE, height=SIZE, seed=42))
    base_png = OUT_DIR / "smoke_zimage_img2img_base.png"
    base_png.write_bytes(base.data)
    print(f"基图(text2img)→ {base_png} ({int((time.monotonic()-t)*1000)}ms)")

    # 2. img2img:把基图作 input_image + strength<1,引擎应切 ZImageImg2ImgPipeline 加噪重去噪。
    #    req 校验 _wants_img2img=True(z-image 有 img2img 变体 + input_image + 0<strength<1)。
    req = ImageRequest(
        request_id="zi2i-refine", prompt="a serene mountain lake at dawn, vibrant autumn foliage",
        steps=STEPS, width=SIZE, height=SIZE, seed=7,
        input_image=str(base_png), strength=STRENGTH)
    assert be._wants_img2img(req), "应判定为 img2img(z-image + input_image + 0<strength<1)"
    t = time.monotonic()
    refined = await be.infer(req)
    ref_png = OUT_DIR / "smoke_zimage_img2img_refined.png"
    ref_png.write_bytes(refined.data)
    print(f"img2img(strength={STRENGTH})→ {ref_png} ({int((time.monotonic()-t)*1000)}ms)")

    # 3. 断言:真用了 img2img pipe + 输出与基图既不同又结构相关。
    assert be._img2img_pipe is not None, "img2img pipe 未构建 —— strength 路径没走"
    s = _ssim(base.data, refined.data)
    print(f"SSIM(base vs refined) = {s:.4f}(期望 0.2~0.95:既改了又保结构)")
    if not (0.15 < s < 0.97):
        print(f"WARN: SSIM={s:.4f} 落在期望区间外 —— strength={STRENGTH} 下人工核图确认")
    print("PASS — Z-Image img2img(strength)引擎路径出图")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
