"""PR-1 真模型 smoke(standalone,需 GPU)—— 细粒度图整模型单卡出图。

不进 pytest 默认套件;手动跑(dev_env_gotchas:真模型 standalone)。镜像 runner
_node_executor 的真实路径:get_or_load_image_adapter(整模型→单卡)→ adapter.infer。

用法:
    cd backend
    # A: fp8 落 3090(cuda:0),不 offload
    SMOKE_DEVICE=cuda:0 SMOKE_DTYPE=fp8_e4m3 uv run python tests/manual/smoke_granular_pr1.py
    # B: bf16 落 Pro 6000(cuda:1)—— 先释放该卡的常驻 vLLM!
    SMOKE_DEVICE=cuda:1 SMOKE_DTYPE=default  uv run python tests/manual/smoke_granular_pr1.py
    # C: A + LoRA 串联
    SMOKE_DEVICE=cuda:0 SMOKE_DTYPE=fp8_e4m3 SMOKE_LORA=/media/heygo/Program/models/nous/image/loras/klein_9B_Turbo_r128.safetensors \
        uv run python tests/manual/smoke_granular_pr1.py

环境变量:
    SMOKE_DEVICE   cuda:0 | cuda:1 | cuda:2     (默认 cuda:0)
    SMOKE_DTYPE    default | bfloat16 | fp8_e4m3 (默认 default)
    SMOKE_MODEL    HF-layout 模型根目录          (默认 Flux2-klein-9B)
    SMOKE_LORA     可选 LoRA .safetensors 绝对路径
    SMOKE_PROMPT   提示词                        (默认见下)
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
import time
from pathlib import Path

# standalone 脚本:把 backend/ 加进 sys.path(pytest 会自动加,直接 python 不会)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ROOT = Path(os.environ.get(
    "SMOKE_MODEL",
    "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"))
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:0")
DTYPE = os.environ.get("SMOKE_DTYPE", "default")
LORA = os.environ.get("SMOKE_LORA", "").strip()
PROMPT = os.environ.get("SMOKE_PROMPT", "a photo of a red fox sitting in autumn leaves, sharp focus")
OUT_DIR = Path(__file__).parent / "_smoke_out"


def _rep(component_dir: Path) -> str:
    hits = sorted(glob.glob(str(component_dir / "*.safetensors")))
    if not hits:
        raise SystemExit(f"组件目录无 .safetensors: {component_dir}")
    return hits[0]


async def main() -> None:
    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest, LoRASpec
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())

    loras = []
    if LORA:
        loras = [LoRASpec(name=Path(LORA).stem, path=LORA, strength=0.8)]

    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=_rep(MODEL_ROOT / "transformer"),
                              device=DEVICE, dtype=DTYPE, adapter_arch="flux2", loras=loras),
        "clip": ComponentSpec(kind="clip", file=_rep(MODEL_ROOT / "text_encoder"),
                              device=DEVICE, dtype=DTYPE),
        "vae":  ComponentSpec(kind="vae", file=_rep(MODEL_ROOT / "vae"),
                              device=DEVICE, dtype=DTYPE),
    }

    print(f"[smoke] device={DEVICE} dtype={DTYPE} lora={LORA or '-'}")
    print(f"[smoke] model_root={MODEL_ROOT}")
    t0 = time.monotonic()
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    print(f"[smoke] adapter loaded in {time.monotonic() - t0:.1f}s")

    req = ImageRequest(
        request_id="smoke-1", prompt=PROMPT, negative_prompt="",
        width=1024, height=1024, steps=20, cfg_scale=4.0, seed=42,
        components=components, pipeline_class="Flux2KleinPipeline",
    )
    t1 = time.monotonic()
    result = await adapter.infer(req)
    dt = time.monotonic() - t1
    print(f"[smoke] infer done in {dt:.1f}s media_type={result.media_type} bytes={len(result.data or b'')}")

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{DEVICE.replace(':', '')}_{DTYPE}{'_lora' if LORA else ''}"
    out = OUT_DIR / f"smoke_{tag}.png"
    out.write_bytes(result.data)
    print(f"[smoke] saved → {out}")
    print(f"[smoke] meta={result.metadata}")
    print(f"[smoke] OK device={DEVICE} dtype={DTYPE} infer={dt:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
