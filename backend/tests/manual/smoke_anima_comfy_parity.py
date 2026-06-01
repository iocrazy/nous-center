"""对齐 ComfyUI 的 anima workflow 参数,验 nous-center AnimaPipeline 是否复现噪点。

ComfyUI workflow(用户给的 ainma Workflow.json):
  UNETLoader  anima/anima-base-v1.0.safetensors
  CLIPLoader  qwen_3_06b_base.safetensors  type=qwen_image
  VAELoader   qwen_image_vae.safetensors
  KSampler    seed=325014996338511 steps=40 cfg=4 euler sgm_uniform denoise=1
  EmptyLatent 896×1152 (非正方!)
  + 完整 positive/negative prompt

standalone(绕过 runner),落 cuda:1=Pro6000。出图正常 → 噪点是 runner 路径 bug;
出图也噪点 → pipeline/参数 bug(尺寸非方 / sgm_uniform 调度差异)。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANIMA_WEIGHTS = "/media/heygo/Program/models/nous/image/diffusion_models/anima/anima-base-v1.0.safetensors"
QWEN_WEIGHTS = "/media/heygo/Program/models/nous/image/text_encoders/qwen_3_06b_base.safetensors"
QWEN_TOKENIZER = "/home/heygo/sites/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
T5_TOKENIZER = "/home/heygo/sites/ComfyUI/comfy/text_encoders/t5_tokenizer"  # LLMAdapter 桥接必需
VAE_WEIGHTS = "/media/heygo/Program/models/nous/image/vae/qwen_image_vae.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"

# 完全照搬 ComfyUI workflow 的 prompt + 参数
POSITIVE = ("masterpiece, best quality, score_9, score_8, score_7, safe, highres, newest, "
            "@silvana, 1girl, solo, dragon girl, monster girl, young adult woman, stunning anime beauty, "
            "elegant and slightly wild, long flowing hair, luminous eyes, dragon horns, "
            "delicate scale details on skin, dragon tail, fantasy creature design, "
            "standing in a shallow river, splashing water around her, crystal-clear stream")
NEGATIVE = ("worst quality, low quality, score_1, score_2, score_3, lowres, blurry, jpeg artifacts, "
            "noisy, overexposed, underexposed, bad anatomy, bad proportions, bad hands, text, logo, watermark")


def main() -> None:
    import torch  # noqa: PLC0415
    from src.services.inference.arch_anima import AnimaPipeline  # noqa: PLC0415

    print(f"[parity] loading on {DEVICE}...")
    t0 = time.time()
    pipe = AnimaPipeline.from_components(
        anima_weights=ANIMA_WEIGHTS, qwen_weights=QWEN_WEIGHTS,
        qwen_tokenizer_dir=QWEN_TOKENIZER, vae_weights=VAE_WEIGHTS,
        t5_tokenizer_dir=T5_TOKENIZER,  # 启用 LLMAdapter 桥接(否则 DiT 收原始 qwen 隐状态 → 噪点)
        device=DEVICE, dtype=torch.bfloat16,
    )
    print(f"  ✓ pipe ready ({time.time()-t0:.1f}s)")

    OUT_DIR.mkdir(exist_ok=True)
    # ComfyUI 完全相同参数:896×1152, steps=40, cfg=4, seed=325014996338511
    cases = [
        ("comfy_parity_896x1152", 896, 1152, 40, 4.0, 325014996338511),
        # 对照:正方 1024(UI 默认),看尺寸是否是因素
        ("comfy_parity_1024sq", 1024, 1024, 40, 4.0, 325014996338511),
    ]
    for label, w, h, steps, cfg, seed in cases:
        print(f"\n[parity] {label}: {w}x{h} steps={steps} cfg={cfg} seed={seed}")
        t0 = time.time()
        img = pipe(
            prompt=POSITIVE, negative_prompt=NEGATIVE,
            num_inference_steps=steps, width=w, height=h,
            seed=seed, guidance_scale=cfg,
        )
        out = OUT_DIR / f"smoke_{label}.png"
        img.save(out)
        print(f"  ✓ saved {out.name} ({img.size})  {time.time()-t0:.1f}s")

    print("\n[parity] done — 肉眼看 _smoke_out/smoke_comfy_parity_*.png 是否噪点")


if __name__ == "__main__":
    main()
