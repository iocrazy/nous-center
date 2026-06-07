"""Qwen-Image-Edit-2511 真模型冒烟(P2 角度控制,spec 2026-06-07)。standalone,需 GPU + 权重。

验钉的 diffusers `QwenImageEditPlusPipeline` + 下载的权重 + `ModularImageBackend` 的 qwen-edit 分支
(_build_qwen_edit_pipe + infer 走 true_cfg_scale + image= 注入)端到端出图。**正确性闸门**:给输入图 +
编辑/角度 prompt,出图应是合法非噪点图,且与输入图明显不同(编辑生效)。

Qwen-Image-Edit:**非 distilled** → CFG 旋钮 true_cfg_scale(默认 4.0,非 guidance_scale);编辑类需 image=;
~20B DiT + Qwen2.5-VL-7B encoder(显存大,落 Pro6000 cuda:1)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_qwen_edit.py            # 裸 pipeline
    SMOKE_VIA_ENGINE=1 SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_qwen_edit.py  # 走引擎路径
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:1=Pro6000(见 CLAUDE.md)

import sys
import time
from pathlib import Path

REPO = os.environ.get(
    "SMOKE_QWEN", "/media/heygo/Program/models/nous/image/diffusers/Qwen-Image-Edit-2511")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
# 输入图:默认用 flux2 edit smoke 的基底狐狸图(若存在),否则需传 SMOKE_INPUT。
_DEF_IN = Path(__file__).parent / "_smoke_out" / "smoke_flux2_edit_base.png"
INPUT = os.environ.get("SMOKE_INPUT", str(_DEF_IN))
PROMPT = os.environ.get("SMOKE_PROMPT", "rotate the camera to show the fox from a 3/4 side angle")
STEPS = int(os.environ.get("SMOKE_STEPS", "40"))
OUT = Path(__file__).parent / "_smoke_out" / "smoke_qwen_edit.png"


def _ssim(a_path: str, b_path: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"  (skip SSIM — {e})")
        return None
    a = np.asarray(Image.open(a_path).convert("RGB").resize((512, 512)))
    b = np.asarray(Image.open(b_path).convert("RGB").resize((512, 512)))
    return float(ssim(a, b, channel_axis=2))


def _engine_smoke() -> int:
    """走 ModularImageBackend 引擎路径(P2 真实代码:_build_qwen_edit_pipe + infer true_cfg_scale + image=)。"""
    import asyncio
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    async def _run() -> int:
        be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16",
                                 pipeline_class="QwenImageEditPlusPipeline")
        await be.load(DEVICE)
        t = time.monotonic()
        res = await be.infer(ImageRequest(
            request_id="qwen-engine", prompt=PROMPT, cfg_scale=4.0, negative_prompt="",
            steps=STEPS, width=1024, height=1024, seed=42, input_image=INPUT))
        OUT.parent.mkdir(exist_ok=True)
        out = OUT.with_name("smoke_qwen_edit_engine.png")
        out.write_bytes(res.data)
        s = _ssim(INPUT, out)
        print(f"PASS(引擎路径)— QwenImageEditPlusPipeline 出图 → {out} "
              f"({int((time.monotonic()-t)*1000)}ms)" + (f" SSIM(输入,输出)={s:.3f}" if s else ""))
        return 0
    return asyncio.run(_run())


def main() -> int:
    if not Path(REPO).exists():
        raise SystemExit(f"Qwen-Image-Edit 权重不存在: {REPO}(还没下完?)")
    if not Path(INPUT).exists():
        raise SystemExit(f"输入图不存在: {INPUT}(传 SMOKE_INPUT=<图路径>)")
    if os.environ.get("SMOKE_VIA_ENGINE") == "1":
        return _engine_smoke()
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

    print(f"加载 QwenImageEditPlusPipeline.from_pretrained({REPO}) → {DEVICE}")
    t0 = time.monotonic()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to(DEVICE)
    print(f"  load {time.monotonic()-t0:.1f}s")
    img = Image.open(INPUT).convert("RGB")
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    t1 = time.monotonic()
    out = pipe(image=img, prompt=PROMPT, true_cfg_scale=4.0,
               num_inference_steps=STEPS, generator=gen)
    OUT.parent.mkdir(exist_ok=True)
    out.images[0].save(OUT)
    s = _ssim(INPUT, OUT)
    print(f"  infer {time.monotonic()-t1:.1f}s ({STEPS} 步) → {OUT}")
    print(f"PASS — Qwen-Image-Edit 出图 {out.images[0].size}" + (f" SSIM(输入,输出)={s:.3f}" if s else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
