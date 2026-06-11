"""dpm++ SDE follow-up 真机:Z-Image dpmpp_2m_sde / dpmpp_3m_sde(对照 ComfyUI k_diffusion
sample_dpmpp_2m_sde / sample_dpmpp_3m_sde 的 CONST/RF 分支)。

验:① 两个 SDE 采样器各出图正常(合法 PNG,非噪点);② 各 ≠ euler(SSIM<0.99 → 生效);
③ 同 seed 复现字节一致(噪声走 seeded randn,确定性);④ 2m_sde ≠ 3m_sde(不同阶)。
噪声用 seeded randn 代 BrownianTree —— 单向前传相邻区间布朗增量本就独立单位正态,分布等价;
故对齐口径 = 分布正确 + 可复现,非逐字节对齐 ComfyUI。零回归看 smoke_zimage_split golden。

用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_dpmpp_sde.py
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
        return ImageRequest(request_id=f"sde-{sampler}", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=STEPS, width=1024, height=1024, seed=SEED, sampler_name=sampler)

    r_eu = await be.infer(_req("euler"))
    p_eu = _save(r_eu.data, "zsde_euler.png")
    print(f"  [euler]               {len(r_eu.data)//1024}KB")
    ok = ok and len(r_eu.data) > 10000 and r_eu.media_type == "image/png"

    imgs = {}
    for sampler in ("dpmpp_2m_sde", "dpmpp_3m_sde"):
        r = await be.infer(_req(sampler))
        p = _save(r.data, f"zsde_{sampler}.png")
        imgs[sampler] = p
        valid = len(r.data) > 10000 and r.media_type == "image/png"
        print(f"  [{sampler:<19}] {len(r.data)//1024}KB -> {p}  {'✓' if valid else 'FAIL'}")
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

    s2 = _ssim(imgs["dpmpp_2m_sde"], imgs["dpmpp_3m_sde"])
    if s2 is not None:
        print(f"  ④ SSIM(dpmpp_2m_sde, dpmpp_3m_sde)={s2:.4f} → {'不同 ✓' if s2 < 0.99 else 'FAIL 几乎一样'}")
        ok = ok and s2 < 0.99

    be.unload()
    print(f"\n{'✅ smoke_zimage_dpmpp_sde PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
