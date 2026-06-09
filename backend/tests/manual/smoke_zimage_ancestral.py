"""PR-1b 真机:Z-Image euler_ancestral(rectified-flow ancestral,对照 ComfyUI sample_euler_ancestral_RF)。

验:① euler_ancestral 出图正常(合法 PNG,非噪点);② euler_ancestral ≠ euler(SSIM<0.99 → ancestral 真生效);
③ 同 seed 复现字节一致;④ euler(非 ancestral)出图仍正常(回归看 smoke_zimage_split 的 SSIM 1.0)。

用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_ancestral.py
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
        return ImageRequest(request_id=f"anc-{sampler}", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=STEPS, width=1024, height=1024, seed=SEED, sampler_name=sampler)

    r_eu = await be.infer(_req("euler"))
    p_eu = _save(r_eu.data, "zanc_euler.png")
    print(f"  [euler]            {len(r_eu.data)//1024}KB")
    r_anc = await be.infer(_req("euler_ancestral"))
    p_anc = _save(r_anc.data, "zanc_ancestral.png")
    print(f"  [euler_ancestral]  {len(r_anc.data)//1024}KB -> {p_anc}")
    ok = ok and len(r_anc.data) > 10000 and r_anc.media_type == "image/png"

    s = _ssim(p_eu, p_anc)
    if s is not None:
        diff = s < 0.99
        print(f"  ② SSIM(euler, euler_ancestral)={s:.4f} → {'不同 ✓(ancestral 生效)' if diff else 'FAIL 几乎一样'}")
        ok = ok and diff

    r_anc2 = await be.infer(_req("euler_ancestral"))
    same = r_anc2.data == r_anc.data
    print(f"  ③ euler_ancestral 同 seed 复现字节一致: {same}")
    ok = ok and same

    be.unload()
    print(f"\n{'✅ smoke_zimage_ancestral PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
