"""PR-4 SSIM 归因诊断 — 隔离「采样数学」 vs 「跨设备」两个差异源。

三张图(同 prompt+seed+steps):
  P_single : stock Flux2KleinPipeline, 全 cuda:1
  S_single : 我们的 ImageSampler, 全 cuda:1 (get_or_load_image_adapter)
  S_cross  : 我们的 ImageSampler, unet=cuda:1 clip=cuda:0 vae=cuda:2

成对 SSIM:
  SSIM(P_single, S_single) : 采样数学 vs Pipeline,同设备 —— 应 >0.99(复现 PR-2 验收;
                             <0.99 ⇒ Task 6 重构回归了数学,真 bug)
  SSIM(S_single, S_cross)  : 仅设备摆放差异(clip/vae 在 3090 vs Pro6000)—— 隔离硬件方差
  SSIM(P_single, S_cross)  : smoke [4] 那个复合指标
"""
import asyncio
import glob
import io
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.base import ImageRequest
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager

ROOT = os.path.expandvars("$LOCAL_MODELS_PATH/image/diffusers/Flux2-klein-9B")
PROMPT = "a small grey kitten sitting on a wooden table, soft natural lighting"
SEED, STEPS, SIZE = 4242, 9, 512


def _file(sub):
    return sorted(glob.glob(f"{ROOT}/{sub}/*.safetensors"))[0]


def _mm():
    reg = ModelRegistry.__new__(ModelRegistry)
    reg._config_path = ""
    reg._specs = {}
    return ModelManager(registry=reg, allocator=GPUAllocator())


def _arr(x):
    if isinstance(x, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(x)).convert("RGB"))
    return np.array(x.convert("RGB"))


def _comps(u, c, v):
    return {
        "unet": ComponentSpec(kind="unet", adapter_arch="flux2", file=_file("transformer"), device=u, dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", clip_arch="flux2", file=_file("text_encoder"), device=c, dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae", file=_file("vae"), device=v, dtype="bfloat16"),
    }


async def _gen(mm, comps):
    a = await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    r = await a.infer(ImageRequest(request_id="d", prompt=PROMPT, seed=SEED, steps=STEPS, width=SIZE, height=SIZE))
    return r.data


async def main():
    # S_single + S_cross share one mm (transformer base key on cuda:1 reused).
    mm = _mm()
    s_single = await _gen(mm, _comps("cuda:1", "cuda:1", "cuda:1"))
    open("/tmp/pr4_s_single.png", "wb").write(s_single)
    s_cross = await _gen(mm, _comps("cuda:1", "cuda:0", "cuda:2"))
    open("/tmp/pr4_s_cross.png", "wb").write(s_cross)

    # P_single stock pipeline, free after.
    from diffusers import Flux2KleinPipeline
    p = Flux2KleinPipeline.from_pretrained(ROOT, torch_dtype=torch.bfloat16).to("cuda:1")
    gen = torch.Generator(device="cuda:1").manual_seed(SEED)
    p_single = p(prompt=PROMPT, num_inference_steps=STEPS, height=SIZE, width=SIZE, generator=gen).images[0]
    p_single.save("/tmp/pr4_p_single.png")
    del p
    torch.cuda.empty_cache()

    def s(a, b):
        return ssim(_arr(a), _arr(b), channel_axis=2, data_range=255)

    print(f"SSIM(P_single, S_single) [math, same dev]   = {s(p_single, s_single):.4f}  (>0.99 expected; <0.99 = refactor regression)")
    print(f"SSIM(S_single, S_cross)  [device only]       = {s(s_single, s_cross):.4f}  (isolates 3090-vs-Pro6000 variance)")
    print(f"SSIM(P_single, S_cross)  [composite/smoke#4] = {s(p_single, s_cross):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
