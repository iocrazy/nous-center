"""真机:Ideogram-4 双 DiT 单文件 **fp8 + DiT offload=cpu 塞 24G 3090**(产品派发路径)。需 GPU。

实施验证(spec 2026-06-13 offload 支持):DiT loader 选 fp8_e4m3 + offload=cpu(双 DiT cpu-stash 轮转)、
TE 驻卡(offload=none)。经 get_or_load_image_adapter → comp_offloads={transformer:cpu,
unconditional_transformer:cpu, text_encoder:none, vae:none} → _place_components_per_device 把双 DiT 挂
cpu-stash hook、TE/VAE 驻卡 → fp8 量化 → 峰值 TE(8)+1DiT(9.3)+VAE ≈ ~18G **入 24G 3090**。

判据:① 在 24G 卡(cuda:2)出连贯图不 OOM;② peak < 23G(真省到 3090 能跑)。
对照:offload=none bf16 是 53G(只能大卡);本路是 fp8+DiT 轮转塞小卡的真价值。
用法:cd backend && SMOKE_DEVICE=cuda:2 uv run python tests/manual/smoke_ideogram4_offload_fit.py
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

    dev = torch.device(DEV)
    print(f"[ideo-fit] {torch.cuda.get_device_properties(dev).name} ({torch.cuda.get_device_properties(dev).total_memory/1024**3:.0f}G) device={DEV}")
    torch.cuda.reset_peak_memory_stats(dev)

    # DiT: fp8 + offload=cpu(cpu-stash 轮转);TE/VAE: 驻卡(offload=none)。
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=DIT, device=DEV, dtype="fp8_e4m3",
                                          adapter_arch="ideogram4", unconditional_file=DIT_U, offload="cpu", loras=[]),
        "clip": ComponentSpec(kind="clip", file=TE, device=DEV, dtype="bfloat16", offload="none"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEV, dtype="bfloat16", offload="none"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    print("[ideo-fit] get_or_load(fp8 + 双 DiT cpu-stash + TE 驻卡)…")
    adapter = await mm.get_or_load_image_adapter(components, "Ideogram4Pipeline")
    co = adapter.comp_offloads
    print(f"[ideo-fit] comp_offloads={co}  (transformer/unconditional 应 cpu,text_encoder/vae 应 none)")
    ok = co.get("transformer") == "cpu" and co.get("unconditional_transformer") == "cpu" and co.get("text_encoder") == "none"

    res = await adapter.infer(ImageRequest(
        request_id="ideo-fit", prompt="a photo of a red fox in autumn leaves, sharp focus",
        steps=STEPS, width=SIZE, height=SIZE, seed=42))
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    Path("/tmp/ideo_offload_fit.png").write_bytes(res.data)
    img_ok = _valid_png(res.data)
    fit_ok = peak < 23000  # 真入 24G 3090
    print(f"[ideo-fit] 出图 ({len(res.data)//1024}KB) peak_vram={peak:.0f}MB "
          f"{'✓连贯' if img_ok else 'FAIL'} {'✓入24G(<23G)' if fit_ok else 'FAIL 峰值超'}")
    ok = ok and img_ok and fit_ok
    print(f"\n{'✅ smoke PASS:fp8+DiT offload 双 DiT 单文件塞 24G 3090 出图' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
