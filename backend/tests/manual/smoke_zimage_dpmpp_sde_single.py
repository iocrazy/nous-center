"""dpm++ SDE 收官真机:Z-Image dpmpp_sde(单步二阶随机,对照 ComfyUI sample_dpmpp_sde CONST 分支)。

验:① 出图正常(合法 PNG,非噪点);② ≠ euler(SSIM<0.99 → 生效);③ 同 seed 复现字节一致
(噪声 seeded randn,确定性);④ ≠ dpmpp_2m_sde(不同算法)。dpmpp_sde NFE 翻倍(中间多一次前向)。
零回归看 smoke_zimage_split golden。

用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_dpmpp_sde_single.py
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
        return ImageRequest(request_id=f"sde1-{sampler}", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=STEPS, width=1024, height=1024, seed=SEED, sampler_name=sampler)

    p_eu = _save((await be.infer(_req("euler"))).data, "zsde1_euler.png")
    p_2msde = _save((await be.infer(_req("dpmpp_2m_sde"))).data, "zsde1_2msde.png")

    r = await be.infer(_req("dpmpp_sde"))
    p = _save(r.data, "zsde1_dpmpp_sde.png")
    valid = len(r.data) > 10000 and r.media_type == "image/png"
    print(f"  [dpmpp_sde] {len(r.data)//1024}KB -> {p}  {'✓' if valid else 'FAIL'}")
    ok = ok and valid

    s = _ssim(p_eu, p)
    if s is not None:
        print(f"  ② SSIM(euler, dpmpp_sde)={s:.4f} → {'不同 ✓(生效)' if s < 0.99 else 'FAIL 几乎一样'}")
        ok = ok and s < 0.99

    same = (await be.infer(_req("dpmpp_sde"))).data == r.data
    print(f"  ③ dpmpp_sde 同 seed 复现字节一致: {same}")
    ok = ok and same

    s2 = _ssim(p_2msde, p)
    if s2 is not None:
        print(f"  ④ SSIM(dpmpp_2m_sde, dpmpp_sde)={s2:.4f} → {'不同 ✓' if s2 < 0.99 else 'FAIL 几乎一样'}")
        ok = ok and s2 < 0.99

    be.unload()
    print(f"\n{'✅ smoke_zimage_dpmpp_sde_single PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
