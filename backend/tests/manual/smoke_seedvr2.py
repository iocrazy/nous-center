"""SeedVR2 引擎真模型 smoke —— 验 SeedVR2UpscaleBackend 经我们 adapter 真出超分图。

standalone,落 cuda:1=Pro 6000。CLAUDE.md:改引擎前必跑真模型 smoke。
PR-2 验证:adapter.load() + upscale() 走 NumZ 4 阶段,256→1024 出清晰超分(非噪点/崩)。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_seedvr2.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
MODEL_DIR = os.environ.get("SEEDVR2_MODEL_DIR", "/media/heygo/Program/models/nous/image/SEEDVR2")
OUT_DIR = Path(__file__).parent / "_smoke_out"
# 输入:一张清晰图缩小,验超分细节重建(默认用 scheduler smoke 出的狐狸图)。
SRC = os.environ.get(
    "SMOKE_SRC",
    str(Path(__file__).parent / "_smoke_out" / "sched_kl_optimal.png"),
)


def main() -> None:
    from PIL import Image  # noqa: PLC0415

    from src.services.inference.image_seedvr2 import SeedVR2UpscaleBackend  # noqa: PLC0415

    OUT_DIR.mkdir(exist_ok=True)
    src = Image.open(SRC).convert("RGB")
    small = src.resize((256, 256))
    small.save(OUT_DIR / "seedvr2_engine_in.png")
    print(f"[seedvr2] input {small.size} → upscale on {DEVICE}")

    be = SeedVR2UpscaleBackend(model_dir=MODEL_DIR, device=DEVICE)
    t0 = time.monotonic()
    be.load()
    print(f"  ✓ load {time.monotonic()-t0:.1f}s")

    t1 = time.monotonic()
    out = be.upscale(small, resolution=1024, seed=42)
    print(f"  ✓ upscale {time.monotonic()-t1:.1f}s → {out.size}")
    out.save(OUT_DIR / "seedvr2_engine_out.png")
    print("[seedvr2] done — 肉眼看 _smoke_out/seedvr2_engine_out.png 是否清晰超分")


if __name__ == "__main__":
    main()
