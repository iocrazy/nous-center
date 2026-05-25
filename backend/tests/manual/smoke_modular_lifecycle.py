"""PR-1 T5 — ComponentsManager 长活 runner 生命周期(自查 P1)。standalone,需 GPU。

runner 是长进程、连续服务多请求。验:同一 ModelManager(modular 引擎,持单个
ComponentsManager)连续 N 次 infer,显存**不单调上涨**(无 per-request 泄漏)、不 OOM。
combo cache 命中后应稳定。

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 SMOKE_DTYPE=default uv run python tests/manual/smoke_modular_lifecycle.py
"""
from __future__ import annotations

import asyncio
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ROOT = Path(os.environ.get(
    "SMOKE_MODEL", "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"))
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
DTYPE = os.environ.get("SMOKE_DTYPE", "default")
N = int(os.environ.get("SMOKE_N", "5"))


def _rep(d: Path) -> str:
    hits = sorted(glob.glob(str(d / "*.safetensors")))
    if not hits:
        raise SystemExit(f"组件目录无 .safetensors: {d}")
    return hits[0]


def _vram_mb(device: str) -> int | None:
    idx = device.split(":")[-1] if ":" in device else "0"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=" + idx, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


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
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=_rep(MODEL_ROOT / "transformer"),
                              device=DEVICE, dtype=DTYPE, adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=_rep(MODEL_ROOT / "text_encoder"), device=DEVICE, dtype=DTYPE),
        "vae":  ComponentSpec(kind="vae", file=_rep(MODEL_ROOT / "vae"), device=DEVICE, dtype=DTYPE),
    }
    print(f"[life] engine=modular device={DEVICE} dtype={DTYPE} N={N}")
    print(f"[life] VRAM start: {_vram_mb(DEVICE)}MB")

    vram = []
    for i in range(N):
        adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
        req = ImageRequest(
            request_id=f"life-{i}", prompt=f"a red fox, take {i}", negative_prompt="",
            width=768, height=768, steps=12, cfg_scale=4.0, seed=42 + i,
            components=components, pipeline_class="Flux2KleinPipeline",
        )
        t = time.monotonic()
        await adapter.infer(req)
        used = _vram_mb(DEVICE)
        vram.append(used)
        print(f"[life] iter {i}: infer {time.monotonic()-t:.1f}s | VRAM {used}MB")

    print(f"\n[life] VRAM 序列: {vram}")
    # 暖机后(第 2 次起)应稳定:后半段最大-最小 < 500MB 视为无泄漏
    tail = [v for v in vram[1:] if v is not None]
    if len(tail) >= 2:
        drift = max(tail) - min(tail)
        verdict = "无泄漏 ✅" if drift < 500 else f"⚠️ 漂移 {drift}MB(查泄漏/双记账)"
        print(f"[life] 暖机后 VRAM 漂移 {drift}MB → {verdict}")
    print("[life] OK")


if __name__ == "__main__":
    asyncio.run(main())
