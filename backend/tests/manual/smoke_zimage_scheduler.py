"""PR-1(Z-Image 引擎地基)真机:Z-Image 调度器放开(simple/beta/…)。standalone,需 GPU。

验:① normal text2img 仍走整段 pipe(零回归,出图正常);② simple/beta 经手写循环出图正常(合法 PNG);
③ simple/beta 出图 ≠ normal(SSIM < 0.99 → 证调度器真生效,不是没接);④ 同 scheduler 同 seed 可复现。

改 image_modular.py 必另跑 smoke_image_ab.py(Flux2 golden)+ smoke_zimage_split.py(normal 分段 SSIM 1.0)。

用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_zimage_scheduler.py
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

    def _req(sched: str) -> ImageRequest:
        return ImageRequest(request_id=f"sch-{sched}", prompt=PROMPT, negative_prompt="", cfg_scale=0.0,
                            steps=STEPS, width=1024, height=1024, seed=SEED, scheduler=sched)

    imgs = {}
    for sched in ("normal", "simple", "beta"):
        r = await be.infer(_req(sched))
        assert r.media_type == "image/png", f"{sched}: {r.media_type}"
        p = _save(r.data, f"zsched_{sched}.png")
        imgs[sched] = p
        print(f"  [{sched:6s}] 出图 {len(r.data)//1024}KB -> {p}")
        ok = ok and len(r.data) > 10000

    # ③ simple/beta 应与 normal 不同(调度器真生效)
    for sched in ("simple", "beta"):
        s = _ssim(imgs["normal"], imgs[sched])
        if s is not None:
            diff = s < 0.99
            print(f"  ③ SSIM(normal, {sched})={s:.4f} → {'不同 ✓(调度器生效)' if diff else 'FAIL 几乎一样(没接上?)'}")
            ok = ok and diff

    # ④ 同 scheduler 同 seed 复现(simple 再来一次,字节一致)
    r2 = await be.infer(_req("simple"))
    same = r2.data == imgs["simple"].read_bytes()
    print(f"  ④ simple 同 seed 复现字节一致: {same}")
    ok = ok and same

    be.unload()
    print(f"\n{'✅ smoke_zimage_scheduler PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
