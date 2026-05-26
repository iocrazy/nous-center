"""PR-anima-7 真模型 e2e:anima 30 步 + CFG 4.5 + 1024×1024 真出图。

按 anima README 推荐配置:30-50 steps,CFG 4-5。这是产品质量验证 —
出图能看 / 不糊 / 跟 anime 风格对应。SSIM ≥ 0.95 对照 ComfyUI 留 future
(需 ComfyUI 端也跑同 prompt + seed)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_anima_pr7.py

cuda:1 Pro 6000 预计 30-60s。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANIMA_WEIGHTS = "/media/heygo/Program/models/nous/image/diffusion_models/anima/anima-base-v1.0.safetensors"
QWEN_WEIGHTS = "/media/heygo/Program/models/nous/image/text_encoders/qwen_3_06b_base.safetensors"
QWEN_TOKENIZER = "/home/heygo/sites/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
VAE_WEIGHTS = "/media/heygo/Program/models/nous/image/vae/qwen_image_vae.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"

# anima README 推荐:positive prefix + danbooru tags + 通用 negative。
POSITIVE_PREFIX = "masterpiece, best quality, score_7, safe, "
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, deformed"


def main() -> None:
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import AnimaPipeline  # noqa: PLC0415

    for label, p in [("anima", ANIMA_WEIGHTS), ("qwen", QWEN_WEIGHTS),
                     ("qwen_tokenizer", QWEN_TOKENIZER), ("vae", VAE_WEIGHTS)]:
        if not Path(p).exists():
            print(f"[anima-pr7] !! {label} missing: {p} — skip")
            return

    _ = torch.zeros(1, device=DEVICE)
    torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))

    print(f"[anima-pr7] loading on {DEVICE}...")
    t0 = time.time()
    pipe = AnimaPipeline.from_components(
        anima_weights=ANIMA_WEIGHTS,
        qwen_weights=QWEN_WEIGHTS,
        qwen_tokenizer_dir=QWEN_TOKENIZER,
        vae_weights=VAE_WEIGHTS,
        device=DEVICE, dtype=torch.bfloat16,
    )
    print(f"  ✓ pipe ready({time.time()-t0:.1f}s)")

    # 出 3 张图:cond-only / cfg=4.5+neg / 大 1024 真尺寸
    test_cases = [
        # (label, prompt, negative, cfg, steps, size)
        ("cond_4step_256", "a cute fox, 1girl, anime style", "", 1.0, 4, 256),
        ("cfg_30step_512", POSITIVE_PREFIX + "1girl, fox ears, brown hair, smile",
         NEGATIVE, 4.5, 30, 512),
        ("cfg_30step_1024_seed42", POSITIVE_PREFIX + "1girl, fox ears, brown hair, smile",
         NEGATIVE, 4.5, 30, 1024),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    for label, prompt, neg, cfg, steps, size in test_cases:
        print(f"\n[anima-pr7] {label}: size={size}, steps={steps}, cfg={cfg}, "
              f"neg={'Y' if neg else '-'}")
        torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))
        t0 = time.time()
        img = pipe(
            prompt=prompt, negative_prompt=neg,
            num_inference_steps=steps, width=size, height=size, seed=42,
            guidance_scale=cfg,
        )
        latency = time.time() - t0
        peak = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / 1024**2
        out_path = OUT_DIR / f"smoke_anima_pr7_{label}.png"
        img.save(out_path)
        print(f"  ✓ saved {out_path.name}({img.size})  "
              f"latency={latency:.1f}s  peak={peak:.0f}MiB ({peak/1024:.2f}GiB)")

    print("\n[anima-pr7] verdict = PASS(3 个 case 出图;质量主观看 + 留 SSIM vs ComfyUI 工作流)")


if __name__ == "__main__":
    main()
