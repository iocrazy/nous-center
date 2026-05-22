"""PR-1 T4 — 引擎 A/B 受控对比(D6:查清 6.4s vs 27s)。standalone,需 GPU。

**单引擎单进程**(避免两引擎模型同卡共存 OOM):每次跑一个引擎,存 ab_<engine>.png +
打印 load/infer 耗时。两次跑完(legacy + modular,完全相同输入)再单独算 SSIM。

用法(落 cuda:1 Pro 6000):
    cd backend
    SMOKE_ENGINE=legacy  SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_image_ab.py
    SMOKE_ENGINE=modular SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_image_ab.py
    # 再比:
    SMOKE_SSIM=1 uv run python tests/manual/smoke_image_ab.py   # 比 ab_legacy.png vs ab_modular.png
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ROOT = Path(os.environ.get(
    "SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"))
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
DTYPE = os.environ.get("SMOKE_DTYPE", "bfloat16")  # 强制 bf16(对齐 spike,避免 default→fp32 OOM)
ENGINE = os.environ.get("SMOKE_ENGINE", "legacy")
PROMPT = os.environ.get("SMOKE_PROMPT", "a photo of a red fox sitting in autumn leaves, sharp focus, detailed")
OUT_DIR = Path(__file__).parent / "_smoke_out"
SEED, STEPS, SIZE = 42, 20, 1024


def _rep(d: Path) -> str:
    hits = sorted(glob.glob(str(d / "*.safetensors")))
    if not hits:
        raise SystemExit(f"组件目录无 .safetensors: {d}")
    return hits[0]


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"[ab] SSIM 跳过(缺 skimage/PIL: {e})")
        return None
    a = np.asarray(Image.open(p1).convert("L"))
    b = np.asarray(Image.open(p2).convert("L"))
    return float(ssim(a, b))


async def _run(mm, components, engine: str) -> tuple[float, float, Path]:
    from src.services.inference.base import ImageRequest

    os.environ["NOUS_IMAGE_ENGINE"] = engine
    t0 = time.monotonic()
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    load_s = time.monotonic() - t0

    req = ImageRequest(
        request_id=f"ab-{engine}", prompt=PROMPT, negative_prompt="",
        width=SIZE, height=SIZE, steps=STEPS, cfg_scale=4.0, seed=SEED,
        components=components, pipeline_class="Flux2KleinPipeline",
    )
    t1 = time.monotonic()
    res = await adapter.infer(req)
    infer_s = time.monotonic() - t1

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"ab_{engine}.png"
    out.write_bytes(res.data)
    print(f"[ab:{engine}] load {load_s:.1f}s | infer {infer_s:.1f}s → {out}")
    return load_s, infer_s, out


async def main() -> None:
    # SSIM-only 模式:比已存的两图,不加载模型
    if os.environ.get("SMOKE_SSIM"):
        a, b = OUT_DIR / "ab_legacy.png", OUT_DIR / "ab_modular.png"
        if not (a.exists() and b.exists()):
            raise SystemExit(f"缺 {a} 或 {b}(先各跑一次 legacy/modular)")
        s = _ssim(a, b)
        print(f"[ab] SSIM(legacy, modular) = {s:.4f}" if s is not None else "[ab] SSIM n/a")
        return

    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    components = {
        "unet": ComponentSpec(kind="unet", file=_rep(MODEL_ROOT / "transformer"),
                              device=DEVICE, dtype=DTYPE, adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=_rep(MODEL_ROOT / "text_encoder"), device=DEVICE, dtype=DTYPE),
        "vae":  ComponentSpec(kind="vae", file=_rep(MODEL_ROOT / "vae"), device=DEVICE, dtype=DTYPE),
    }
    print(f"[ab] engine={ENGINE} device={DEVICE} dtype={DTYPE} seed={SEED} steps={STEPS} size={SIZE}")
    await _run(mm, components, ENGINE)
    print("[ab] OK")


if __name__ == "__main__":
    asyncio.run(main())
