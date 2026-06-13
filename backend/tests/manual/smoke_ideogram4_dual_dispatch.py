"""Ideogram-4 双 DiT 单文件 —— 经**生产派发路径**(get_or_load_image_adapter)出图。需 GPU。PR-3。

验 PR-3 runner/组件管线:diffusion_models ComponentSpec 携带 `unconditional_file`(合并节点
ideogram4_dual_guider 挂的第二 DiT)→ get_or_load_image_adapter → _is_standalone_single_file →
build_bridged_transformer(cond)+ build_bridged_transformer(config_sub=unconditional_transformer,uncond)
+ build_bridged_text_encoder(Qwen3-VL)+ build_bridged_vae → ModularImageBackend(4 override)→ infer。

对照 PR-1 的 smoke_ideogram4_singlefile(直接构造 backend),本 smoke 走**真实派发入口**(runner 同款),
覆盖 PR-3 的 unconditional_file 透传 + L1/combo 键含第二 DiT。

判据(spec 2026-06-12):fp8 单文件对 bf16 整模型 SSIM 到不了 0.9,故判「出连贯图 + image_count=1」。

用法:cd backend && SMOKE_DEVICE=cuda:1 SMOKE_SIZE=512 uv run python tests/manual/smoke_ideogram4_dual_dispatch.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PKG = "/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/models"
DIT = f"{PKG}/diffusion_models/ideogram4_fp8_scaled.safetensors"
DIT_U = f"{PKG}/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
TE = f"{PKG}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
VAE = f"{PKG}/vae/flux2-vae.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "12"))


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

    dev = torch.device(DEVICE)
    print(f"[ideo-dispatch] {torch.cuda.get_device_properties(dev).name} device={DEVICE} size={SIZE}")
    torch.cuda.reset_peak_memory_stats(dev)

    # DTYPE=fp8_e4m3 + OFFLOAD=cpu → 双 DiT fp8 + cpu-stash 轮转(只卸 DiT、TE/VAE 驻卡)→ 塞 24G 3090。
    dtype = os.environ.get("DTYPE", "bfloat16")
    # 关键:cond DiT 的 ComponentSpec 携带 unconditional_file(= PR-2 合并节点透传给 runner 的形态)。
    components = {
        "diffusion_models": ComponentSpec(
            kind="diffusion_models", file=DIT, device=DEVICE, dtype=dtype,
            adapter_arch="ideogram4", unconditional_file=DIT_U, loras=[]),
        "clip": ComponentSpec(kind="clip", file=TE, device=DEVICE, dtype=dtype),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    offload = os.environ.get("OFFLOAD", "none")  # cpu = 卡紧时逐组件流(峰值=单组件)
    print(f"[ideo-dispatch] dtype={dtype} offload={offload}")
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    print(f"[ideo-dispatch] get_or_load_image_adapter(Ideogram4Pipeline, offload={offload}) — 派发建双 DiT override…")
    adapter = await mm.get_or_load_image_adapter(components, "Ideogram4Pipeline", offload=offload)
    # 校验第二 DiT override 真被建出来(PR-3 派发路径)。
    uncond = getattr(adapter, "_unconditional_transformer_override", "MISSING")
    print(f"[ideo-dispatch] adapter={type(adapter).__name__} repo={adapter.repo} "
          f"uncond_override={'有' if uncond not in (None, 'MISSING') else uncond}")
    ok = uncond not in (None, "MISSING")

    res = await adapter.infer(ImageRequest(
        request_id="ideo-dispatch",
        prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
        steps=STEPS, width=SIZE, height=SIZE, seed=42))
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    out = Path("/tmp/ideo_dual_dispatch.png")
    out.write_bytes(res.data)
    img_ok = _valid_png(res.data) and res.usage.image_count == 1
    ok = ok and img_ok
    print(f"[ideo-dispatch] saved → {out} ({len(res.data)//1024}KB) image_count={res.usage.image_count} "
          f"peak_vram={peak:.0f}MB {'✓连贯' if img_ok else 'FAIL'}")
    print(f"\n{'✅ smoke_ideogram4_dual_dispatch PASS(派发路径双 DiT 单文件出图)' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
