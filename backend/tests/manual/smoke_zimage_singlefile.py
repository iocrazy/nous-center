"""PR-2 真机闸门:Z-Image 单文件分开载入(build_bridged_* via from_single_file + 装配)。standalone,需 GPU。

验:用真下载的 comfy 单文件(z_image_turbo_bf16 UNet + qwen_3_4b 编码器 + ae VAE)经 build_bridged_*
(z-image 走 diffusers from_single_file)装配成 ZImagePipeline,与整模型 from_pretrained 出图对比。
单文件 == 整模型同权重 → SSIM 应 ≥0.95(证分开载入装配正确,不是没接/接错)。

改 image_modular.py 必另跑 smoke_image_ab.py(Flux2 golden)+ smoke_zimage_split.py(分段 SSIM 1.0)。
用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_singlefile.py
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
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
UNET = f"{ROOT}/diffusion_models/z_image_turbo_bf16.safetensors"
ENC = f"{ROOT}/text_encoders/qwen_3_4b.safetensors"
AE = f"{ROOT}/vae/ae.safetensors"
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"


def _save(data: bytes, name: str) -> Path:
    p = Path(tempfile.gettempdir()) / name
    p.write_bytes(data)
    return p


def _ssim(a: Path, b: Path):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(np.asarray(Image.open(a).convert("L")), np.asarray(Image.open(b).convert("L"))))


async def main() -> int:
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_modular import (
        ModularImageBackend,
        build_bridged_text_encoder,
        build_bridged_transformer,
        build_bridged_vae,
    )
    from src.services.model_manager import _reference_repo_for_arch

    repo = _reference_repo_for_arch("z-image")
    print(f"参考库(z-image): {repo}")
    assert repo, "z-image 参考库未解析(_reference_repo_for_arch 返回 None)"
    ok = True

    def _req(rid: str) -> ImageRequest:
        return ImageRequest(request_id=rid, prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=8, width=1024, height=1024, seed=42)

    # ---- 单文件分开载入:build_bridged_*(z-image 走 from_single_file)----
    print("① build_bridged_transformer(z_image_turbo_bf16 单文件)…")
    t_ov = build_bridged_transformer(
        ComponentSpec(kind="diffusion_models", file=UNET, device=DEV, dtype="bfloat16", adapter_arch="z-image"), repo, DEV)
    print(f"   OK in_channels={t_ov.config.in_channels}")
    print("② build_bridged_vae(ae 单文件,Flux1 VAE)…")
    v_ov = build_bridged_vae(ComponentSpec(kind="vae", file=AE, device=DEV, dtype="bfloat16"), repo, DEV)
    print(f"   OK latent_channels={v_ov.config.latent_channels}")
    print("③ build_bridged_text_encoder(qwen_3_4b 单文件)…")
    c_ov = build_bridged_text_encoder(ComponentSpec(kind="clip", file=ENC, device=DEV, dtype="bfloat16"), repo, DEV)
    print("   OK text_encoder built")

    be_sf = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="ZImagePipeline",
                                transformer_override=t_ov, text_encoder_override=c_ov, vae_override=v_ov)
    await be_sf.load(DEV)
    r_sf = await be_sf.infer(_req("sf"))
    p_sf = _save(r_sf.data, "zsf_singlefile.png")
    print(f"④ 单文件装配出图 -> {p_sf} ({len(r_sf.data)//1024}KB)")
    ok = ok and r_sf.media_type == "image/png" and len(r_sf.data) > 10000
    be_sf.unload()

    # ---- 整模型基线 ----
    be_full = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be_full.load(DEV)
    r_full = await be_full.infer(_req("full"))
    p_full = _save(r_full.data, "zsf_fullmodel.png")
    print(f"⑤ 整模型基线出图 -> {p_full} ({len(r_full.data)//1024}KB)")
    be_full.unload()

    s = _ssim(p_sf, p_full)
    gate = s >= 0.95
    ok = ok and gate
    print(f"\n  ★ 闸门 SSIM(单文件分开载入, 整模型) = {s:.4f} {'PASS ✓(≥0.95,同权重装配正确)' if gate else 'FAIL ✗'}")

    print(f"\n{'✅ smoke_zimage_singlefile PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
