"""SeedVR2 PR-3a 真模型 smoke —— 验**经 ModelManager 路径** + InferenceAdapter ABC 出图。

跟 smoke_seedvr2.py(直接 SeedVR2UpscaleBackend.upscale 引擎核心)不同,本 smoke 走
PR-3a 接入路径:
    ModelManager.get_or_load_seedvr2_adapter() → adapter.infer(UpscaleRequest) → PNG bytes
验:① by-key 装载注册进 _models(LRU/快照可见)② infer 接 UpscaleRequest(base64 data URI
输入图)③ 二次调命中缓存(同 model_id)。

standalone,落 cuda:1=Pro 6000。CLAUDE.md:改引擎/接入前必跑真模型 smoke。
用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_seedvr2_modelmanager.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
MODEL_DIR = os.environ.get("SEEDVR2_MODEL_DIR", "/media/heygo/Program/models/nous/image/SEEDVR2")
OUT_DIR = Path(__file__).parent / "_smoke_out"
SRC = os.environ.get(
    "SMOKE_SRC",
    str(Path(__file__).parent / "_smoke_out" / "sched_kl_optimal.png"),
)


async def main() -> None:
    from PIL import Image  # noqa: PLC0415

    from src.services.gpu_allocator import GPUAllocator  # noqa: PLC0415
    from src.services.inference.base import UpscaleRequest  # noqa: PLC0415
    from src.services.inference.registry import ModelRegistry  # noqa: PLC0415
    from src.services.model_manager import ModelManager  # noqa: PLC0415

    OUT_DIR.mkdir(exist_ok=True)
    src = Image.open(SRC).convert("RGB")
    small = src.resize((256, 256))
    # 编成 base64 data URI(模拟 runner 从签名 image_url 取图后塞进 UpscaleRequest.image)。
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    print(f"[seedvr2-mm] input {small.size} → infer via ModelManager on {DEVICE}")

    mm = ModelManager(registry=ModelRegistry("configs/models.yaml"), allocator=GPUAllocator())

    t0 = time.monotonic()
    adapter = await mm.get_or_load_seedvr2_adapter(model_dir=MODEL_DIR, device=DEVICE)
    print(f"  ✓ get_or_load_seedvr2_adapter {time.monotonic()-t0:.1f}s  is_loaded={adapter.is_loaded}")

    snap = mm.loaded_models_snapshot()
    print(f"  ✓ _models 快照 {len(snap)} 条: {[s['model_id'] for s in snap]}")
    assert any(s["model_id"].startswith("image:SeedVR2:") for s in snap), "SeedVR2 未登记进 _models"

    req = UpscaleRequest(request_id="smoke-1", image=data_uri, resolution=1024, seed=42)
    t1 = time.monotonic()
    res = await adapter.infer(req)
    print(f"  ✓ infer {time.monotonic()-t1:.1f}s → {res.media_type} {len(res.data)}B meta={res.metadata}")
    out = Image.open(io.BytesIO(res.data))
    out.save(OUT_DIR / "seedvr2_mm_out.png")
    print(f"  ✓ 出图 {out.size} → _smoke_out/seedvr2_mm_out.png")

    # 二次调:同参数应命中缓存(同 model_id,不重新装载)。
    adapter2 = await mm.get_or_load_seedvr2_adapter(model_dir=MODEL_DIR, device=DEVICE)
    assert adapter2 is adapter, "二次 get_or_load 没命中缓存(应同一 adapter 实例)"
    print("  ✓ 二次 get_or_load 命中缓存(同 adapter 实例)")
    print("[seedvr2-mm] done — 肉眼看 _smoke_out/seedvr2_mm_out.png 是否清晰超分")


if __name__ == "__main__":
    asyncio.run(main())
