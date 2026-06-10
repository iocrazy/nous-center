"""PR-2 真机闸门:LCS 锐化干预(vendor 光栅 PCA 标定 + per-step post-CFG hook)。standalone,需 GPU。

验 lcs_sharpness 干预端到端:
  ① interventions=[lcs_sharpness strength=2] → 出图比基线**更锐**(Laplacian 方差↑)+ 与基线不同 + 有效。
  ② strength=0 → edit_vec=0 → 出图 **byte-identical** 基线(零回归)。
  ③ strength=2 复现一致(确定性 + 标定缓存命中)。
标定(正弦光栅 PCA,~448 VAE encode)首次跑、缓存到 image/lcs_cache/sharpness_<vae指纹>.safetensors。

改 image_modular.py 必另跑 smoke_image_ab.py + smoke_zimage_split.py。
用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_lcs_sharpness.py
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = "/media/heygo/Program/models/nous/image"
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
UNET = f"{ROOT}/diffusion_models/z_image_turbo_bf16.safetensors"
ENC = f"{ROOT}/text_encoders/qwen_3_4b.safetensors"
AE = f"{ROOT}/vae/ae.safetensors"
PROMPT = "a photorealistic portrait of a woman, autumn park, soft light, detailed skin"


def _lap_var(png: bytes) -> float:
    """Laplacian 方差(锐度代理:越大越锐)。"""
    import io

    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float64)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(im, (3, 3))
    lap = (w * k).sum(axis=(-1, -2))
    return float(lap.var())


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
    assert repo, "z-image 参考库未解析"
    t_ov = build_bridged_transformer(
        ComponentSpec(kind="diffusion_models", file=UNET, device=DEV, dtype="bfloat16", adapter_arch="z-image"), repo, DEV)
    v_ov = build_bridged_vae(ComponentSpec(kind="vae", file=AE, device=DEV, dtype="bfloat16"), repo, DEV)
    c_ov = build_bridged_text_encoder(ComponentSpec(kind="clip", file=ENC, device=DEV, dtype="bfloat16"), repo, DEV)
    be = ModularImageBackend(repo=repo, device=DEV, dtype="bfloat16", pipeline_class="ZImagePipeline",
                             transformer_override=t_ov, text_encoder_override=c_ov, vae_override=v_ov)
    await be.load(DEV)

    def _req(iv):
        return ImageRequest(request_id="lcs", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=8, width=768, height=768, seed=42,
                            sampler_name="euler", scheduler="simple", interventions=iv)

    def _h(b: bytes) -> str:
        return hashlib.md5(b).hexdigest()

    sharp = [{"_type": "lcs_sharpness", "strength": 2.0, "start_step": 0, "end_step": 8}]
    zero = [{"_type": "lcs_sharpness", "strength": 0.0, "start_step": 0, "end_step": 8}]

    print("① 基线(无干预)…")
    r_base = await be.infer(_req(None))
    print("② lcs_sharpness strength=2(首次触发光栅标定,稍慢)…")
    r_sharp = await be.infer(_req(sharp))
    print("③ lcs_sharpness strength=0(应 no-op)…")
    r_zero = await be.infer(_req(zero))
    print("④ strength=2 复现…")
    r_sharp2 = await be.infer(_req(sharp))
    be.unload()

    Path("/tmp/lcs_base.png").write_bytes(r_base.data)
    Path("/tmp/lcs_sharp.png").write_bytes(r_sharp.data)
    lv_base, lv_sharp = _lap_var(r_base.data), _lap_var(r_sharp.data)
    gate_sharper = lv_sharp > lv_base                       # 锐度真上去
    gate_changed = _h(r_sharp.data) != _h(r_base.data)
    gate_zero = _h(r_zero.data) == _h(r_base.data)          # strength=0 = no-op(零回归)
    gate_repro = _h(r_sharp.data) == _h(r_sharp2.data)      # 确定性 + 缓存命中
    print(f"\n  ① 锐度 Laplacian 方差: base={lv_base:.1f} sharp={lv_sharp:.1f} → 更锐: {gate_sharper}")
    print(f"  ② 改变出图: {gate_changed}")
    print(f"  ③ strength=0 byte-identical(零回归): {gate_zero}")
    print(f"  ④ 确定性复现: {gate_repro}")
    ok = gate_sharper and gate_changed and gate_zero and gate_repro
    print(f"\n{'✅ smoke_lcs_sharpness PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
