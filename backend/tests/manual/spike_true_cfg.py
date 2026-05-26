"""SPIKE — 证伪「nous 把 true-cfg 模型 (True-v2) 强制跑成蒸馏 → cfg/negative 被丢」。

根因(config 级已坐实):单文件 True-v2 借 diffusers/Flux2-klein-9B 的 model_index.json
(is_distilled:true)当 config 参考 → modular 走蒸馏 block(guidance=None、无 negative 分支)→
cfg/negative 全失效。上个 session cfg_verify「cfg SSIM=1.0」错判成「模型蒸馏」,实为「管线把 cfg 扔了」。

本 spike 用**标准** Flux2KleinPipeline(is_distilled=False)装 True-v2 单文件三组件(经 nous 生产桥接
build_bridged_*),同 prompt/seed/steps **扫 cfg**:
  - cfg=1.0  → do_cfg=False → 单次前向(≈ 现蒸馏基线)
  - cfg=3.5/5/7 → true-CFG(空 negative,纯 CFG)
  - cfg=4 + 自定义 negative(预编码 negative_prompt_embeds)
SSIM(cfg1, cfgX):若 << 1.0 且 cfg~3.5-5 出图更锐/更贴 prompt → 假设成立,nous 该按模型类别
(diffusion_models/ = comfy true-cfg)走 true-cfg 路径、透传 cfg+negative(对齐 ComfyUI)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/spike_true_cfg.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
VAE = f"{IMG}/vae/flux2-vae.safetensors"
REPO = f"{IMG}/diffusers/Flux2-klein-9B"  # 仅借 config/tokenizer/scheduler
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT = Path(__file__).parent / "_smoke_out"

PROMPT = ("a photo of a red fox sitting in autumn leaves, sharp focus, highly detailed, "
          "professional photography, golden hour lighting")
NEG = "blurry, low quality, distorted, deformed, washed out, oversaturated, jpeg artifacts"
SEED, STEPS, SIZE = 42, 25, 1024


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"[spike] SSIM 跳过(缺 skimage/PIL: {e})")
        return None
    a = np.asarray(Image.open(p1).convert("L"))
    b = np.asarray(Image.open(p2).convert("L"))
    return float(ssim(a, b))


def main() -> None:
    import torch
    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_modular import (
        build_bridged_text_encoder,
        build_bridged_transformer,
        build_bridged_vae,
    )

    dev = torch.device(DEVICE)
    print(f"[spike] device={DEVICE} {torch.cuda.get_device_properties(dev).name}")
    OUT.mkdir(exist_ok=True)

    unet_spec = ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                              dtype="bfloat16", adapter_arch="flux2")
    clip_spec = ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16")
    vae_spec = ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16")

    t0 = time.monotonic()
    print("[spike] 桥接 transformer (True-v2 bf16)…")
    transformer = build_bridged_transformer(unet_spec, REPO, DEVICE)
    print("[spike] 桥接 text_encoder (qwen3 fp8mixed)…")
    text_encoder = build_bridged_text_encoder(clip_spec, REPO, DEVICE)
    print("[spike] 桥接 vae…")
    vae = build_bridged_vae(vae_spec, REPO, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    print(f"[spike] 组件就绪 {time.monotonic()-t0:.1f}s")

    pipe = Flux2KleinPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, transformer=transformer, is_distilled=False,
    )

    def gen() -> "torch.Generator":
        return torch.Generator(device=DEVICE).manual_seed(SEED)

    runs: list[tuple[str, float, object]] = [
        ("cfg1.0", 1.0, None),       # do_cfg=False → 单次前向 = 蒸馏基线
        ("cfg3.5", 3.5, None),       # true-CFG 空 neg
        ("cfg5.0", 5.0, None),
        ("cfg7.0", 7.0, None),
    ]

    # 自定义 negative:预编码 → negative_prompt_embeds
    neg_embeds = pipe.encode_prompt(prompt=NEG, device=dev)[0]

    paths: dict[str, Path] = {}
    for tag, cfg, _ in runs:
        t1 = time.monotonic()
        out = pipe(prompt=PROMPT, num_inference_steps=STEPS, width=SIZE, height=SIZE,
                   guidance_scale=cfg, generator=gen())
        p = OUT / f"truecfg_{tag}.png"
        out.images[0].save(p)
        paths[tag] = p
        print(f"[spike] {tag:8} cfg={cfg} → {p.name} ({time.monotonic()-t1:.1f}s)")

    # cfg=4 + 自定义 negative
    t1 = time.monotonic()
    out = pipe(prompt=PROMPT, num_inference_steps=STEPS, width=SIZE, height=SIZE,
               guidance_scale=4.0, negative_prompt_embeds=neg_embeds, generator=gen())
    p = OUT / "truecfg_cfg4_neg.png"
    out.images[0].save(p)
    paths["cfg4_neg"] = p
    print(f"[spike] cfg4_neg cfg=4 +neg → {p.name} ({time.monotonic()-t1:.1f}s)")

    print("\n[spike] === SSIM vs cfg=1.0 基线(越低=cfg 影响越大)===")
    base = paths["cfg1.0"]
    for tag, p in paths.items():
        if tag == "cfg1.0":
            continue
        s = _ssim(base, p)
        print(f"[spike]   SSIM(cfg1.0, {tag:8}) = {s:.4f}" if s is not None else f"  {tag} n/a")

    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"\n[spike] peak_vram={peak:.0f}MB. 看图:{OUT}/truecfg_*.png")
    print("[spike] 判读:SSIM 明显<1 且 cfg3.5~5 更锐/更贴 prompt → True-v2 是 true-cfg,"
          "nous 蒸馏强制是质量/控制差的根因。")


if __name__ == "__main__":
    main()
