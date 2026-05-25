"""fp8 紧凑加载 —— 经**生产路径**(ModularImageBackend dtype=fp8_e4m3)在 3090 出图。需 GPU。

验 Load Diffusion Model 选 bf16 整模型 + weight_dtype=fp8 的真实链路:
  ModularImageBackend(dtype="fp8_e4m3") → _ensure_pipe(load bf16 → torchao weight-only 量化
  transformer+text_encoder)→ infer。验:出正确图 + 进 24GB 3090 + 显存 < bf16。

对照 spike_quant_compact.py(直接调 torchao);本 smoke 走生产 adapter,验接线正确。

用法:
    cd backend
    SMOKE_DEVICE=cuda:2 uv run python tests/manual/smoke_fp8_compact.py     # 3090
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:0/2=3090, cuda:1=Pro6000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:2")
OUT_DIR = Path(__file__).parent / "_smoke_out"


async def main() -> None:
    import torch

    from src.services.inference.base import ImageRequest
    from src.services.inference.image_modular import ModularImageBackend

    dev = torch.device(DEVICE)
    props = torch.cuda.get_device_properties(dev)
    print(f"[fp8-compact] device={DEVICE} ({props.name} {props.total_memory/1024**3:.1f}GB) sm={torch.cuda.get_device_capability(dev)}")
    torch.cuda.reset_peak_memory_stats(dev)

    # 生产路径:dtype=fp8_e4m3 → _ensure_pipe 内部 torchao weight-only 量化
    be = ModularImageBackend(repo=REPO, device=DEVICE, dtype="fp8_e4m3")
    res = await be.infer(ImageRequest(
        request_id="fp8-compact",
        prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
        steps=20, width=1024, height=1024, seed=42,
    ))
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2

    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / "smoke_fp8_compact.png"
    out_png.write_bytes(res.data)
    print(f"[fp8-compact] saved → {out_png} ({len(res.data)} bytes) meta={res.metadata}")
    print(f"[fp8-compact] peak_vram={peak:.0f}MB latency_ms={res.usage.latency_ms}")
    print("[fp8-compact] OK — fp8 weight-only 经生产 ModularImageBackend 出图")


if __name__ == "__main__":
    asyncio.run(main())
