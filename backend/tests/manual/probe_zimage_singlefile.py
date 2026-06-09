"""PR-2 探针:确认下载的 Z-Image comfy 单文件能经 diffusers from_single_file 加载(不手写键转换)。

回答:① ZImageTransformer2DModel.from_single_file(z_image_turbo_bf16) 成功?② AutoencoderKL.from_single_file(ae)?
③ 装配成 ZImagePipeline 出图正确?用 cuda:1。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch

ROOT = "/media/heygo/Program/models/nous/image"
REPO = f"{ROOT}/diffusers/Z-Image-Turbo"  # 参考库(tokenizer/scheduler/config)
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
UNET = f"{ROOT}/diffusion_models/z_image_turbo_bf16.safetensors"
AE = f"{ROOT}/vae/ae.safetensors"

from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel

print("① transformer from_single_file …")
tf = ZImageTransformer2DModel.from_single_file(UNET, torch_dtype=torch.bfloat16, config=f"{REPO}/transformer")
print(f"   OK in_channels={tf.config.in_channels} dtype={tf.dtype}")

print("② vae from_single_file …")
try:
    vae = AutoencoderKL.from_single_file(AE, torch_dtype=torch.bfloat16)
    print(f"   OK latent_channels={vae.config.latent_channels}")
except Exception as e:  # noqa: BLE001
    print(f"   from_single_file 失败:{type(e).__name__}: {e}; 退回 from_pretrained 整模型 vae")
    vae = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).vae

print("③ 装配 ZImagePipeline(单文件 transformer/vae + 整模型 text_encoder/tokenizer)…")
base = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
pipe = ZImagePipeline(
    transformer=tf, vae=vae, text_encoder=base.text_encoder,
    tokenizer=base.tokenizer, scheduler=base.scheduler)
pipe.to(DEV)
g = torch.Generator(DEV).manual_seed(42)
img = pipe(prompt="a red fox in autumn leaves, sharp focus", num_inference_steps=8,
           guidance_scale=0.0, width=1024, height=1024, generator=g).images[0]
img.save("/tmp/probe_zimage_sf.png")
print(f"   出图 -> /tmp/probe_zimage_sf.png size={img.size}")
print("探针完成。")
