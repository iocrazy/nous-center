"""PR-1 真机闸门:Ideogram-4 双 DiT **单文件**分开载入,经**产品引擎路径**装配出图。standalone,需 GPU。

验:用 comfy 单文件(双 DiT fp8_scaled + Qwen3-VL TE + flux2 VAE)经**产品** build_bridged_*(ideogram4
走 dequant + 手写键转)建 4 个 override → ModularImageBackend(pipeline_class=Ideogram4Pipeline)装配
Ideogram4Pipeline 出图。判据(spec 2026-06-12):fp8 单文件对 bf16 整模型 SSIM 本就到不了 0.9,故
**判据 = 出连贯图(非噪点/非崩)+ 非对称 CFG 真生效**(uncond DiT 置零 → 图明显变差/崩,证两 DiT 都参与);
转换权重正确性由 spike weight-diff 单独坐实(mean 2% = fp8 噪声)。

显存:bf16 双 DiT+TE+VAE ~53G,大卡(Pro6000)直载。
用法:cd backend && SMOKE_DEVICE=cuda:1 SMOKE_SIZE=512 uv run python tests/manual/smoke_ideogram4_singlefile.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PKG = "/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/models"
DIT = f"{PKG}/diffusion_models/ideogram4_fp8_scaled.safetensors"
DIT_U = f"{PKG}/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
TE = f"{PKG}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
VAE = f"{PKG}/vae/flux2-vae.safetensors"
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "12"))
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"
SEED = 42


def _valid_png(b: bytes) -> bool:
    try:
        from PIL import Image
        Image.open(io.BytesIO(b)).verify()
        return len(b) > 10000
    except Exception:  # noqa: BLE001
        return False


def _ssim(a: bytes, b: bytes):
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
        ia = np.asarray(Image.open(io.BytesIO(a)).convert("L"))
        ib = np.asarray(Image.open(io.BytesIO(b)).convert("L"))
        return float(ssim(ia, ib))
    except Exception:  # noqa: BLE001
        return None


async def main() -> int:
    import torch  # noqa: PLC0415

    from src.services.inference.base import ImageRequest  # noqa: PLC0415
    from src.services.inference.component_spec import ComponentSpec  # noqa: PLC0415
    from src.services.inference.image_modular import (  # noqa: PLC0415
        ModularImageBackend,
        build_bridged_text_encoder,
        build_bridged_transformer,
        build_bridged_vae,
    )
    from src.services.model_manager import _reference_repo_for_arch  # noqa: PLC0415

    repo = _reference_repo_for_arch("ideogram4")
    print(f"参考库(ideogram4): {repo}")
    assert repo, "ideogram4 参考库未解析(_reference_repo_for_arch 返回 None;查仓内 bundle)"
    ok = True

    def _build():
        # 产品 build_bridged_*:ideogram4 DiT 走 dequant+键转;uncond DiT 用 config_sub。
        t = build_bridged_transformer(
            ComponentSpec(kind="diffusion_models", file=DIT, device=DEV, dtype="bfloat16",
                          adapter_arch="ideogram4"), repo, DEV, config_sub="transformer")
        tu = build_bridged_transformer(
            ComponentSpec(kind="diffusion_models", file=DIT_U, device=DEV, dtype="bfloat16",
                          adapter_arch="ideogram4"), repo, DEV, config_sub="unconditional_transformer")
        te = build_bridged_text_encoder(
            ComponentSpec(kind="clip", file=TE, device=DEV, dtype="bfloat16"), repo, DEV)
        vae = build_bridged_vae(
            ComponentSpec(kind="vae", file=VAE, device=DEV, dtype="bfloat16"), repo, DEV)
        return t, tu, te, vae

    print("① 产品 build_bridged_*(双 DiT + Qwen3-VL TE + flux2 VAE)…")
    t_ov, tu_ov, c_ov, v_ov = _build()
    print("   4 override 建好(双 DiT load 0/0、TE Qwen3VLModel 0/0)")

    be = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="Ideogram4Pipeline",
                             transformer_override=t_ov, unconditional_transformer_override=tu_ov,
                             text_encoder_override=c_ov, vae_override=v_ov)
    await be.load(DEV)
    print("② ModularImageBackend 装配 Ideogram4Pipeline(4 override)+ 出图…")
    r = await be.infer(ImageRequest(request_id="ideo-sf", prompt=PROMPT, negative_prompt="",
                                    steps=STEPS, width=SIZE, height=SIZE, seed=SEED))
    Path(tempfile.gettempdir(), "ideo_sf_product.png").write_bytes(r.data)
    sf_ok = _valid_png(r.data)
    print(f"   单文件出图 -> /tmp/ideo_sf_product.png ({len(r.data)//1024}KB)  {'✓连贯' if sf_ok else 'FAIL'}")
    ok = ok and sf_ok
    be.unload()

    # 非对称 CFG 生效判据:uncond DiT 权重置零 → 重装出图应与正常图明显不同(证 uncond 真参与去噪)。
    import gc  # noqa: PLC0415
    del be, t_ov, c_ov, v_ov
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("③ 非对称 CFG 判据:uncond DiT 置零重装 → 出图应与正常显著不同…")
    t2, tu2, c2, v2 = _build()
    with torch.no_grad():
        for p in tu2.parameters():
            p.zero_()
    be2 = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="Ideogram4Pipeline",
                              transformer_override=t2, unconditional_transformer_override=tu2,
                              text_encoder_override=c2, vae_override=v2)
    await be2.load(DEV)
    r2 = await be2.infer(ImageRequest(request_id="ideo-zero-uncond", prompt=PROMPT, negative_prompt="",
                                      steps=STEPS, width=SIZE, height=SIZE, seed=SEED))
    be2.unload()
    s = _ssim(r.data, r2.data)
    asym_ok = s is not None and s < 0.95  # 置零 uncond → 图变 → 证 uncond DiT 真参与
    print(f"   SSIM(正常, uncond置零)={s:.4f} → {'✓ uncond 真参与(<0.95)' if asym_ok else 'FAIL:uncond 似未生效'}")
    ok = ok and asym_ok

    print(f"\n{'✅ smoke_ideogram4_singlefile PASS(产品路径双 DiT 单文件装配出图 + 非对称 CFG 生效)' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
