"""真模型验 9 个 scheduler 都能出正常图(CLAUDE.md:改 image_modular 前必跑真模型 smoke)。

standalone,落 cuda:1=Pro 6000。每个 scheduler 出一张图,肉眼看是否正常(不噪点/不崩)。
重点验新增的 5 个 injected(simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal)
经 pipe(sigmas=...)注入后真能出图。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ROOT = os.environ.get(
    "SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
SCHEDULERS = ["normal", "karras", "exponential", "beta",
              "sgm_uniform", "simple", "ddim_uniform", "linear_quadratic", "kl_optimal"]


def _rep(sub: str) -> str:
    import glob  # noqa: PLC0415
    hits = sorted(glob.glob(str(Path(MODEL_ROOT) / sub / "*.safetensors")))
    if not hits:
        raise SystemExit(f"缺 {sub}/*.safetensors")
    return hits[0]


async def main() -> None:
    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _Empty(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_Empty(), allocator=GPUAllocator())
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=_rep("transformer"),
                              device=DEVICE, dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=_rep("text_encoder"), device=DEVICE, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae", file=_rep("vae"), device=DEVICE, dtype="bfloat16"),
    }
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[sched] device={DEVICE}, 9 schedulers, 12 steps each")
    for sch in SCHEDULERS:
        req = ImageRequest(
            request_id=f"sched-{sch}", prompt=PROMPT, negative_prompt="",
            width=1024, height=1024, steps=12, cfg_scale=4.0, seed=42,
            components=components, pipeline_class="Flux2KleinPipeline",
            sampler_name="euler", scheduler=sch,
        )
        t0 = time.monotonic()
        try:
            res = await adapter.infer(req)
            out = OUT_DIR / f"sched_{sch}.png"
            out.write_bytes(res.data)
            print(f"  ✓ {sch:18s} {time.monotonic()-t0:.1f}s → {out.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {sch:18s} 失败: {type(e).__name__}: {e}")
    print("[sched] done — 肉眼看 _smoke_out/sched_*.png")


if __name__ == "__main__":
    asyncio.run(main())
