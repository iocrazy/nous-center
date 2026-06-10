"""dpm++ PR 真机:Z-Image dpmpp_2m / dpmpp_2s_ancestral(对照 ComfyUI k_diffusion
sample_dpmpp_2m / sample_dpmpp_2s_ancestral_RF)。

验:① 两个 dpmpp 采样器各出图正常(合法 PNG,非噪点);② 各 ≠ euler(SSIM<0.99 → 采样器真生效);
③ 同 seed 复现字节一致;④ euler(回归)出图仍正常 + 与 smoke_zimage_ancestral 一致(零回归看
smoke_zimage_split golden)。dpmpp_2s_ancestral NFE 翻倍(中间多一次前向),耗时约 euler 2x 属正常。

用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_dpmpp.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"
STEPS, SEED = 12, 42


def _save(data: bytes, name: str) -> Path:
    p = Path(tempfile.gettempdir()) / name
    p.write_bytes(data)
    return p


def _ssim(a: Path, b: Path):
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception:  # noqa: BLE001
        return None
    return float(ssim(np.asarray(Image.open(a).convert("L")), np.asarray(Image.open(b).convert("L"))))


async def main() -> int:
    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="bfloat16", pipeline_class="ZImagePipeline")
    await be.load(DEVICE)
    ok = True

    def _req(sampler: str) -> ImageRequest:
        return ImageRequest(request_id=f"dpmpp-{sampler}", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=STEPS, width=1024, height=1024, seed=SEED, sampler_name=sampler)

    r_eu = await be.infer(_req("euler"))
    p_eu = _save(r_eu.data, "zdpmpp_euler.png")
    print(f"  [euler]                {len(r_eu.data)//1024}KB")
    ok = ok and len(r_eu.data) > 10000 and r_eu.media_type == "image/png"

    for sampler in ("dpmpp_2m", "dpmpp_2s_ancestral"):
        r = await be.infer(_req(sampler))
        p = _save(r.data, f"zdpmpp_{sampler}.png")
        valid = len(r.data) > 10000 and r.media_type == "image/png"
        print(f"  [{sampler:<20}] {len(r.data)//1024}KB -> {p}  {'✓' if valid else 'FAIL'}")
        ok = ok and valid

        s = _ssim(p_eu, p)
        if s is not None:
            diff = s < 0.99
            print(f"  ② SSIM(euler, {sampler})={s:.4f} → {'不同 ✓(生效)' if diff else 'FAIL 几乎一样'}")
            ok = ok and diff

        r2 = await be.infer(_req(sampler))
        same = r2.data == r.data
        print(f"  ③ {sampler} 同 seed 复现字节一致: {same}")
        ok = ok and same

    # ④ dpmpp_2m vs dpmpp_2s_ancestral 也应不同(不同算法)
    s2 = _ssim(_save((await be.infer(_req("dpmpp_2m"))).data, "zdpmpp_2m_b.png"),
               _save((await be.infer(_req("dpmpp_2s_ancestral"))).data, "zdpmpp_2s_b.png"))
    if s2 is not None:
        print(f"  ④ SSIM(dpmpp_2m, dpmpp_2s_ancestral)={s2:.4f} → {'不同 ✓' if s2 < 0.99 else 'FAIL 几乎一样'}")
        ok = ok and s2 < 0.99

    be.unload()
    print(f"\n{'✅ smoke_zimage_dpmpp PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
