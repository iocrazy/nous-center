"""PR-1 真机闸门:采样期 latent 干预挂钩基建(spec 2026-06-10)。standalone,需 GPU。

验 Z-Image 手写循环的 per-step latent 干预挂钩:
  ① interventions=None vs [](空)→ 出图 **byte-identical**(空 chain = no-op,零回归)。
  ② interventions=[test_shift] → 出图与基线**不同**(挂钩真生效、能改 latent)+ 复现一致(确定性)。
test_shift 是 PR-1 管道验证型干预(denoised 沿常向量推);LCS 锐化/色彩在 PR-2/3。

改 image_modular.py 必另跑 smoke_image_ab.py(Flux2 golden)+ smoke_zimage_split.py(分段 SSIM 1.0)。
用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_sampling_intervention.py
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

    def _req(interventions):
        # scheduler=simple → 恒走手写分段循环(挂钩点所在);8 步全程。
        return ImageRequest(request_id="iv", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=8, width=768, height=768, seed=42,
                            sampler_name="euler", scheduler="simple", interventions=interventions)

    def _md5(b: bytes) -> str:
        return hashlib.md5(b).hexdigest()

    print("① 基线(interventions=None)…")
    r_base = await be.infer(_req(None))
    print("② 空 chain(interventions=[])…")
    r_empty = await be.infer(_req([]))
    print("③ test_shift(strength=0.05,全程)…")
    shift = [{"_type": "test_shift", "strength": 0.05, "start_step": 0, "end_step": 100}]
    r_shift = await be.infer(_req(shift))
    print("④ test_shift 复现(同参再跑)…")
    r_shift2 = await be.infer(_req(shift))
    be.unload()

    h_base, h_empty, h_shift, h_shift2 = (_md5(r.data) for r in (r_base, r_empty, r_shift, r_shift2))
    Path("/tmp/iv_base.png").write_bytes(r_base.data)
    Path("/tmp/iv_shift.png").write_bytes(r_shift.data)

    gate_a = h_base == h_empty       # 空 chain = no-op(零回归)
    gate_b = h_shift != h_base       # 干预真生效(改了 latent)
    gate_c = h_shift == h_shift2     # 确定性复现
    print(f"\n  ① 空 chain byte-identical(零回归): {gate_a} (base={h_base[:8]} empty={h_empty[:8]})")
    print(f"  ② test_shift 改变出图: {gate_b} (shift={h_shift[:8]})")
    print(f"  ③ test_shift 确定性复现: {gate_c}")
    ok = gate_a and gate_b and gate_c
    print(f"\n{'✅ smoke_sampling_intervention PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
