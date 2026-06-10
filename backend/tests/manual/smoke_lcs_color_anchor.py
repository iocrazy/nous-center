"""PR-3 真机闸门:LCS 色彩锚定(vendor 色彩 PCA 标定 + per-step 色漂纠正)。standalone,需 GPU。

验 lcs_color_anchor(self_anchor)端到端:
  ① interventions=[lcs_color_anchor intensity=0.8] → 改变出图 + 有效连贯人像。
  ② intensity=0 → 不建闭包 → byte-identical 基线(零回归)。
  ③ intensity=0.8 复现一致(确定性 + 色彩标定缓存命中)。
色彩标定(512 HSV 样本 PCA)首次跑、缓存 image/lcs_cache/lcsdata_<vae指纹>.safetensors。

改 image_modular.py 必另跑 smoke_image_ab.py + smoke_zimage_split.py。
用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_lcs_color_anchor.py
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
        return ImageRequest(request_id="anchor", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=8, width=768, height=768, seed=42,
                            sampler_name="euler", scheduler="simple", interventions=iv)

    def _h(b: bytes) -> str:
        return hashlib.md5(b).hexdigest()

    anchor = [{"_type": "lcs_color_anchor", "mode": "self_anchor", "intensity": 0.8}]
    zero = [{"_type": "lcs_color_anchor", "mode": "self_anchor", "intensity": 0.0}]

    print("① 基线(无干预)…")
    r_base = await be.infer(_req(None))
    print("② lcs_color_anchor intensity=0.8(首次触发色彩标定,稍慢)…")
    r_anchor = await be.infer(_req(anchor))
    print("③ intensity=0(应 no-op)…")
    r_zero = await be.infer(_req(zero))
    print("④ intensity=0.8 复现…")
    r_anchor2 = await be.infer(_req(anchor))
    be.unload()

    Path("/tmp/lcs_anchor_base.png").write_bytes(r_base.data)
    Path("/tmp/lcs_anchor.png").write_bytes(r_anchor.data)
    gate_changed = _h(r_anchor.data) != _h(r_base.data)
    gate_valid = r_anchor.media_type == "image/png" and len(r_anchor.data) > 10000
    gate_zero = _h(r_zero.data) == _h(r_base.data)
    gate_repro = _h(r_anchor.data) == _h(r_anchor2.data)
    print(f"\n  ① 改变出图: {gate_changed}")
    print(f"  ② 出图有效(连贯 PNG): {gate_valid} ({len(r_anchor.data)//1024}KB)")
    print(f"  ③ intensity=0 byte-identical(零回归): {gate_zero}")
    print(f"  ④ 确定性复现: {gate_repro}")
    ok = gate_changed and gate_valid and gate_zero and gate_repro
    print(f"\n{'✅ smoke_lcs_color_anchor PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
