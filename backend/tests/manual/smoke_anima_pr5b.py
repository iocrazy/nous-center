"""PR-anima-5(part 2)真模型 smoke:AnimaPipeline.from_components 完整装配 + 短 denoise + VAE decode。

不验 SSIM(留 PR-7),只验**整 pipeline 不挂** + 输出 PIL Image 形状。
小尺寸(256×256)+ 少步(4)免久等。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_anima_pr5b.py
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


def main() -> None:
    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import AnimaPipeline  # noqa: PLC0415

    for label, p in [("anima", ANIMA_WEIGHTS), ("qwen", QWEN_WEIGHTS),
                     ("qwen_tokenizer", QWEN_TOKENIZER), ("vae", VAE_WEIGHTS)]:
        if not Path(p).exists():
            print(f"[anima-pr5b] !! {label} missing: {p} — skip")
            return

    _ = torch.zeros(1, device=DEVICE)  # CUDA init
    torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))

    print(f"[anima-pr5b] from_components on {DEVICE}...")
    pipe = AnimaPipeline.from_components(
        anima_weights=ANIMA_WEIGHTS,
        qwen_weights=QWEN_WEIGHTS,
        qwen_tokenizer_dir=QWEN_TOKENIZER,
        vae_weights=VAE_WEIGHTS,
        t5_tokenizer_dir=None,  # 不走 LLMAdapter t5xxl 路径
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    print("  ✓ pipe ready (anima DiT + AnimaTextEncoder + VAE + FlowMatchEulerScheduler)")

    # 小尺寸 + 少步,只验 pipeline 流程不挂(SSIM 留 PR-7)。
    prompt = "masterpiece, best quality, a cute fox"
    print(f"[anima-pr5b] pipe(prompt='{prompt[:40]}...', size=256, steps=4, seed=42)")
    img = pipe(prompt=prompt, num_inference_steps=4, width=256, height=256, seed=42)
    assert img.size == (256, 256), f"unexpected image size {img.size}"
    assert img.mode == "RGB"

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "smoke_anima_pr5b_4steps.png"
    img.save(out_path)
    print(f"  ✓ image saved: {out_path}({img.size}, {img.mode})")

    peak_mib = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / 1024**2
    print(f"[anima-pr5b] peak VRAM = {peak_mib:.0f} MiB ({peak_mib/1024:.2f} GiB)")
    print("[anima-pr5b] verdict = PASS(pipeline 装配 + denoise + VAE decode 全跑通)")


if __name__ == "__main__":
    main()
