"""PR-3 真机闸门:Z-Image GGUF 文本编码器端到端出图(CLIPLoaderGGUF 等价)。standalone,需 GPU。

验:Qwen3-4b-Z-Image-Engineer-V4-Q8_0.gguf 经 build_bridged_text_encoder 的 .gguf 分支
(transformers 原生 from_pretrained(gguf_file=))装出编码器 → 装进 ZImagePipeline(turbo 单文件
transformer + ae VAE)→ 真出图。同 pipe 换普通 qwen_3_4b.safetensors 编码器出第二张,
两张都是有效 PNG 且不同(SSIM<1,证实 GGUF Engineer 微调权重真生效,非静默退普通)。

改 image_modular.py 必另跑 smoke_image_ab.py(Flux2 golden)+ smoke_zimage_split.py(分段 SSIM 1.0)。
用法:cd backend && SMOKE_DEVICE=cuda:0 uv run python tests/manual/smoke_zimage_gguf.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = "/media/heygo/Program/models/nous/image"
DEV = os.environ.get("SMOKE_DEVICE", "cuda:0")
UNET = f"{ROOT}/diffusion_models/z_image_turbo_bf16.safetensors"
GGUF = f"{ROOT}/text_encoders/Qwen3-4b-Z-Image-Engineer-V4-Q8_0.gguf"
PLAIN = f"{ROOT}/text_encoders/qwen_3_4b.safetensors"
AE = f"{ROOT}/vae/ae.safetensors"
PROMPT = "a photorealistic portrait of a woman, autumn park, soft light, detailed skin"


def _save(data: bytes, name: str) -> Path:
    p = Path(tempfile.gettempdir()) / name
    p.write_bytes(data)
    return p


def _ssim(a: Path, b: Path) -> float:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(np.asarray(Image.open(a).convert("L")), np.asarray(Image.open(b).convert("L"))))


async def _gen(enc_file: str, repo: str, tag: str) -> Path:
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_modular import (
        ModularImageBackend,
        build_bridged_text_encoder,
        build_bridged_transformer,
        build_bridged_vae,
    )
    t_ov = build_bridged_transformer(
        ComponentSpec(kind="diffusion_models", file=UNET, device=DEV, dtype="bfloat16", adapter_arch="z-image"), repo, DEV)
    v_ov = build_bridged_vae(ComponentSpec(kind="vae", file=AE, device=DEV, dtype="bfloat16"), repo, DEV)
    c_ov = build_bridged_text_encoder(ComponentSpec(kind="clip", file=enc_file, device=DEV, dtype="bfloat16"), repo, DEV)
    be = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="ZImagePipeline",
                             transformer_override=t_ov, text_encoder_override=c_ov, vae_override=v_ov)
    await be.load(DEV)
    r = await be.infer(ImageRequest(request_id=tag, prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                                    steps=8, width=768, height=768, seed=42))
    be.unload()
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    p = _save(r.data, f"zgguf_{tag}.png")
    print(f"  {tag} 出图 -> {p} ({len(r.data)//1024}KB) media={r.media_type}")
    assert r.media_type == "image/png" and len(r.data) > 10000, f"{tag} 出图无效"
    return p


async def main() -> int:
    from src.services.model_manager import _reference_repo_for_arch
    repo = _reference_repo_for_arch("z-image")
    print(f"参考库(z-image): {repo}")
    assert repo, "z-image 参考库未解析"

    print("① GGUF 编码器(Z-Image-Engineer)端到端出图…")
    p_g = await _gen(GGUF, repo, "gguf")
    print("② 普通 qwen_3_4b 编码器出图(对照)…")
    p_p = await _gen(PLAIN, repo, "plain")

    s = _ssim(p_g, p_p)
    # 不同微调编码器 → 出图应不同(SSIM<0.999);但都是有效人像(上面已 assert PNG)。
    diff_ok = s < 0.999
    print(f"\n  ★ SSIM(GGUF, 普通) = {s:.4f} {'PASS ✓(<0.999,GGUF 权重真生效)' if diff_ok else 'FAIL ✗(疑静默退普通)'}")
    print(f"\n{'✅ smoke_zimage_gguf PASS' if diff_ok else '❌ FAIL'}")
    return 0 if diff_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
