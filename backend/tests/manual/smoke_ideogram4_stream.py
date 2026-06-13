"""真机:Ideogram-4 双 DiT 单文件 **offload=stream(lowvram 流式分块)塞 24G 3090**。需 GPU。

验 spec 待做项「offload=stream 模式」:bf16 双 DiT 全驻是 53G,只能大卡;stream 把两个 DiT 都流式分块
(`_apply_stream_offload` 迭代 pipe.components,名字含 "transformer" 的都挂 apply_group_offloading →
第二 DiT 自动覆盖,arch-agnostic),TE/VAE 驻卡 → 峰值 ~TE(16)+流式工作集 → **入 24G 3090**。

判据:① bf16 双 DiT 在 24G 卡出连贯图不 OOM(只有流式真生效才可能,全驻 53G 装不下);② peak < 23G。
对照:cpu offload 是 fp8 轮转(~20G);stream 是 bf16 不量化(对齐整模型 lowvram,spec 2026-06-12)。
用法:cd backend && SMOKE_DEVICE=cuda:2 uv run python tests/manual/smoke_ideogram4_stream.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PKG = "/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/models"
DIT = f"{PKG}/diffusion_models/ideogram4_fp8_scaled.safetensors"
DIT_U = f"{PKG}/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
TE = f"{PKG}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
VAE = f"{PKG}/vae/flux2-vae.safetensors"
DEV = os.environ.get("SMOKE_DEVICE", "cuda:2")
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "8"))


def _valid_png(b: bytes) -> bool:
    try:
        from PIL import Image
        Image.open(io.BytesIO(b)).verify()
        return len(b) > 10000
    except Exception:  # noqa: BLE001
        return False


async def main() -> int:
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

    dev = torch.device(DEV)
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    print(f"[ideo-stream] {torch.cuda.get_device_properties(dev).name} ({total:.0f}G) device={DEV}")
    torch.cuda.reset_peak_memory_stats(dev)

    # 双 DiT offload=stream(bf16,不量化);TE/VAE 驻卡。
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=DIT, device=DEV, dtype="bfloat16",
                                          adapter_arch="ideogram4", unconditional_file=DIT_U, offload="stream", loras=[]),
        "clip": ComponentSpec(kind="clip", file=TE, device=DEV, dtype="bfloat16", offload="none"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEV, dtype="bfloat16", offload="none"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    print("[ideo-stream] get_or_load(双 DiT offload=stream bf16 + TE 驻卡)…")
    adapter = await mm.get_or_load_image_adapter(components, "Ideogram4Pipeline")

    res = await adapter.infer(ImageRequest(
        request_id="ideo-stream", prompt="a photo of a red fox in autumn leaves, sharp focus",
        steps=STEPS, width=SIZE, height=SIZE, seed=42))
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    Path("/tmp/ideo_stream.png").write_bytes(res.data)
    img_ok = _valid_png(res.data)
    fit_ok = peak < 23000  # bf16 双 DiT 全驻 53G;流式真生效才能 <23G 入 24G 卡
    print(f"[ideo-stream] 出图 ({len(res.data)//1024}KB) peak_vram={peak:.0f}MB "
          f"{'✓连贯' if img_ok else 'FAIL'} {'✓入24G(<23G,流式生效)' if fit_ok else 'FAIL 峰值超→流式没覆盖双 DiT'}")
    ok = img_ok and fit_ok
    print(f"\n{'✅ smoke PASS:双 DiT 单文件 offload=stream bf16 塞 24G 3090' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
