"""PR-2 T5 — comfy fp8mixed 单文件经**生产 modular 路径**出图。standalone,需 GPU。

mirrors runner:NOUS_IMAGE_ENGINE=modular + unet=comfy fp8mixed 单文件 + clip/vae=HF →
get_or_load_image_adapter → _get_or_load_modular_adapter(检测 comfy → build_bridged_transformer
→ ModularImageBackend transformer_override)→ infer。验出**正确狐狸图**(非噪声)。

⚠️ 反量化到 bf16 → 不省显存(spec §4)。本 smoke 落 cuda:1(Pro 6000),只验「可用」。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_modular_comfy_fp8.py
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HF_ROOT = Path("/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B")
FP8 = "/media/heygo/Program/models/nous/image/diffusion_models/flux/Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"


def _rep(d: Path) -> str:
    hits = sorted(glob.glob(str(d / "*.safetensors")))
    if not hits:
        raise SystemExit(f"无 .safetensors: {d}")
    return hits[0]


async def main() -> None:
    os.environ["NOUS_IMAGE_ENGINE"] = "modular"
    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    # unet = comfy fp8mixed 单文件;clip/vae = HF(提供 config/scheduler/clip/vae)
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=FP8, device=DEVICE, dtype="bfloat16",
                              adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=_rep(HF_ROOT / "text_encoder"), device=DEVICE, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae", file=_rep(HF_ROOT / "vae"), device=DEVICE, dtype="bfloat16"),
    }
    print(f"[comfy-fp8] engine=modular device={DEVICE}")
    print(f"[comfy-fp8] unet(comfy fp8mixed)={FP8}")
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    print(f"[comfy-fp8] adapter={type(adapter).__name__}")

    req = ImageRequest(
        request_id="comfy-fp8", prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
        negative_prompt="", width=1024, height=1024, steps=20, cfg_scale=4.0, seed=42,
        components=components, pipeline_class="Flux2KleinPipeline",
    )
    res = await adapter.infer(req)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "smoke_modular_comfy_fp8.png"
    out.write_bytes(res.data)
    print(f"[comfy-fp8] saved → {out} ({len(res.data)} bytes) meta={res.metadata}")
    print("[comfy-fp8] OK — comfy fp8mixed 经生产 modular 路径出图")


if __name__ == "__main__":
    asyncio.run(main())
