"""单文件装配 —— 经**生产路径**(get_or_load_image_adapter)在真模型出图。需 GPU。

验 PR-2 产品代码:三组件全单文件 ComponentSpec → get_or_load_image_adapter →
_modular_repo_from_components(无 HF 组件 → _reference_repo_for_arch 架构参考整模型)→
_is_standalone_single_file 三个 True → build_bridged_transformer/text_encoder/vae →
ModularImageBackend(三 override)→ infer。对照 spike(独立脚本),本 smoke 走真实引擎入口。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_single_file_prod.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
VAE = f"{IMG}/vae/flux2-vae.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"


async def main() -> None:
    import torch

    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    dev = torch.device(DEVICE)
    print(f"[sf-prod] {torch.cuda.get_device_properties(dev).name}")  # 触发 CUDA init
    torch.cuda.reset_peak_memory_stats(dev)
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    print(f"[sf-prod] device={DEVICE} 三组件全单文件 → 生产路径")
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    print(f"[sf-prod] adapter={type(adapter).__name__} repo={adapter.repo}")

    res = await adapter.infer(ImageRequest(
        request_id="sf-prod",
        prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
        steps=20, width=1024, height=1024, seed=42))
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "smoke_single_file_prod.png"
    out.write_bytes(res.data)
    print(f"[sf-prod] saved → {out} ({len(res.data)} bytes) peak_vram={peak:.0f}MB "
          f"latency_ms={res.usage.latency_ms}")
    print("[sf-prod] OK — 单文件装配经生产 get_or_load_image_adapter 出图")


if __name__ == "__main__":
    asyncio.run(main())
