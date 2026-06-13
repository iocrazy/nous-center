"""真机:Ideogram-4 双 DiT 单文件 combo **卸载→显存真降**(第二 DiT 不泄漏)。需 GPU。

验 spec 待做项「卸载显存验证」:双 DiT 单文件经 get_or_load 装配后,unload 必须把**两个 DiT**都释放
(对照 [[project_unified_model_mgmt_completion]] 的 combo unload 2G→22G 泄漏:adapter 持 override 引用、
pool 置 None 后 gc 收不掉)。第二 DiT(unconditional_transformer_override)是 adapter 直接持有(非池化)
→ unload 清 override + pipe + gc 应真释放。

判据:加载后显存涨明显(双 DiT+TE+VAE,fp8 offload=none ~27G);unload 后 **回落 ≥ 涨幅的 85%**
(残留 < 15% = 没泄漏第二 DiT)。
用法:cd backend && SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_ideogram4_unload_vram.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PKG = "/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/models"
DIT = f"{PKG}/diffusion_models/ideogram4_fp8_scaled.safetensors"
DIT_U = f"{PKG}/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
TE = f"{PKG}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
VAE = f"{PKG}/vae/flux2-vae.safetensors"
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "8"))


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

    def used_mb() -> float:
        torch.cuda.synchronize(dev)
        return torch.cuda.memory_allocated(dev) / 1024**2

    base = used_mb()
    print(f"[unload] 基线显存={base:.0f}MB device={DEV}")

    # fp8 + offload=none → 双 DiT 都驻 GPU(卸载后降幅反映两个 DiT 是否真释放)。
    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=DIT, device=DEV, dtype="fp8_e4m3",
                                          adapter_arch="ideogram4", unconditional_file=DIT_U, offload="none", loras=[]),
        "clip": ComponentSpec(kind="clip", file=TE, device=DEV, dtype="bfloat16", offload="none"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEV, dtype="bfloat16", offload="none"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Ideogram4Pipeline")
    # 真出一张确保 pipe 实例化(_ensure_pipe 建双 DiT)。
    await adapter.infer(ImageRequest(request_id="unload-warm", prompt="a red fox", steps=STEPS,
                                     width=SIZE, height=SIZE, seed=1))
    loaded = used_mb()
    grew = loaded - base
    print(f"[unload] 加载+出图后显存={loaded:.0f}MB(涨 {grew:.0f}MB)")

    # 卸载该 combo(走 model_manager 真卸载路径)+ adapter.unload(清 override 引用)。
    for mid in list(mm._models.keys()):
        await mm.unload_model(mid)
    adapter.unload()
    gc.collect()
    torch.cuda.empty_cache()
    after = used_mb()
    residual = after - base
    freed_pct = (grew - residual) / grew * 100 if grew > 0 else 0
    print(f"[unload] 卸载后显存={after:.0f}MB(残留 {residual:.0f}MB,释放 {freed_pct:.0f}%)")
    # 残留 < 涨幅 15% = 两个 DiT 都释放,无泄漏。
    ok = grew > 5000 and residual < grew * 0.15
    print(f"\n{'✅ smoke PASS:双 DiT combo 卸载显存真降(第二 DiT 不泄漏)' if ok else '❌ FAIL:残留过多,疑第二 DiT 泄漏'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
