"""Flux2 输入图(编辑/img2img/多参考)真模型冒烟(spec 2026-06-07,input-image plumbing)。

验 `ImageRequest.input_image` → `ModularImageBackend.infer` 把 `image=` 注入 Flux2KleinPipeline,
端到端走真模型出图。**正确性闸门**:同 prompt + 同 seed,带输入图 vs 不带,出图应**明显不同**
(SSIM 低)—— 若 plumbing 断了(image 没真传进 pipe),两图会一模一样。同时两图都得是合法非损坏图。

为何这样验:input-image 是 P2(Qwen-Edit)/ P3(Flux2 多参考编辑)共用的最难跨层管线;Flux2 权重
已在盘 → 现在就能真机验这条管线,不用等 Qwen 下完盲验(见 [[feedback_verify_real_model]])。

用法(Flux2-9B bf16 落 Pro6000 cuda:1):
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_flux2_edit.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:1=Pro6000(见 CLAUDE.md)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
VAE = f"{IMG}/vae/flux2-vae.safetensors"

DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"
BASE_PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
EDIT_PROMPT = os.environ.get("SMOKE_EDIT_PROMPT", "turn it into winter, heavy snow everywhere, snowy forest")
SEED, STEPS, SIZE = 42, 20, 1024


def _ssim(p1: Path, p2: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:  # noqa: BLE001
        print(f"  (skip SSIM — {e})")
        return None
    a = np.asarray(Image.open(p1).convert("RGB"))
    b = np.asarray(Image.open(p2).convert("RGB"))
    return float(ssim(a, b, channel_axis=2))


async def main() -> int:
    from src.services.gpu_allocator import GPUAllocator
    from src.services.inference.base import ImageRequest
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.registry import ModelRegistry
    from src.services.model_manager import ModelManager

    class _EmptyRegistry(ModelRegistry):
        def __init__(self):
            self._config_path = ""
            self._specs = {}

    components = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE,
                                          dtype="bfloat16", adapter_arch="flux2", loras=[]),
        "clip": ComponentSpec(kind="clip", file=CLIP, device=DEVICE, dtype="bfloat16"),
        "vae": ComponentSpec(kind="vae", file=VAE, device=DEVICE, dtype="bfloat16"),
    }
    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    adapter = await mm.get_or_load_image_adapter(components, "Flux2KleinPipeline")
    OUT_DIR.mkdir(exist_ok=True)

    # 1) 先做一张基底图(纯文生图,input_image=None)。
    base = await adapter.infer(ImageRequest(
        request_id="flux2edit-base", prompt=BASE_PROMPT, cfg_scale=4.0, negative_prompt="",
        steps=STEPS, width=SIZE, height=SIZE, seed=SEED))
    base_png = OUT_DIR / "smoke_flux2_edit_base.png"
    base_png.write_bytes(base.data)
    print(f"  [base] 纯文生图 → {base_png.name}")

    # 2) 控制组:用编辑 prompt 纯文生图(无输入图),同 seed —— 作为「没传图」基线。
    control = await adapter.infer(ImageRequest(
        request_id="flux2edit-control", prompt=EDIT_PROMPT, cfg_scale=4.0, negative_prompt="",
        steps=STEPS, width=SIZE, height=SIZE, seed=SEED))
    control_png = OUT_DIR / "smoke_flux2_edit_control_noimg.png"
    control_png.write_bytes(control.data)
    print(f"  [control] 编辑 prompt 纯文生图(无图)→ {control_png.name}")

    # 3) 真编辑:input_image=基底图 + 编辑 prompt,同 seed。
    edit = await adapter.infer(ImageRequest(
        request_id="flux2edit-edit", prompt=EDIT_PROMPT, cfg_scale=4.0, negative_prompt="",
        steps=STEPS, width=SIZE, height=SIZE, seed=SEED, input_image=str(base_png)))
    edit_png = OUT_DIR / "smoke_flux2_edit_edited.png"
    edit_png.write_bytes(edit.data)
    print(f"  [edit] input_image=base + 编辑 prompt → {edit_png.name}")

    # 闸门:带图 vs 不带图(同 prompt/seed)应明显不同 → 证明 image 真传进了 pipe。
    s = _ssim(control_png, edit_png)
    if s is None:
        print("PASS(三图均出,无异常;装 scikit-image 看 SSIM 差异闸门)")
        return 0
    print(f"SSIM(control 无图, edit 带图) = {s:.4f}  ← 应明显 <1(带图改变了输出 = plumbing 通)")
    ok = s < 0.95  # 带图必须实质改变输出;太接近 1 说明 image 没生效
    print("PASS — 输入图真注入 pipe 并改变输出(input-image plumbing 通)" if ok
          else "FAIL — 带图与不带图几乎相同(SSIM≈1)→ input_image 可能没真传进 pipe")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
