"""探针:ZImagePipeline 对 custom sigmas= / latents= 的真实行为(决定 PR-B2 引擎怎么写)。

回答 3 个问题(逐个实测,不猜):
Q1 传 sigmas=[部分,不到 0] → 跑几步?停在哪个噪声水平(部分去噪=leftover)?还是被 append 0 跑到底?
Q2 传 latents=<某 latent> + sigmas=[中段..] → 是原样续采(不重加噪)还是重新缩放/加噪?
Q3 callback_on_step_end 的 cb_kwargs['latents'] 在第 i 步是「该步之后」的 latent?

用法:SMOKE_DEVICE=cuda:1 uv run python tests/manual/probe_zimage_sigmas.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from diffusers import ZImagePipeline

REPO = os.environ.get("SMOKE_ZIMAGE", "/media/heygo/Program/models/nous/image/diffusers/Z-Image-Turbo")
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")
PROMPT = "a red fox in autumn leaves"

pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
pipe.to(DEVICE)
print("init_noise_sigma:", getattr(pipe.scheduler, "init_noise_sigma", "n/a"))

# 记录每步 callback 的 sigma/step
seen = []
def cb(p, i, t, kw):
    sig = float(p.scheduler.sigmas[i]) if hasattr(p.scheduler, "sigmas") else None
    seen.append((i, float(t) if t is not None else None, sig,
                 tuple(kw["latents"].shape) if "latents" in kw else None))
    return kw

# Q1: 传完整 8 步 sigmas(末尾 0)看基线步数
seen.clear()
g = torch.Generator(DEVICE).manual_seed(42)
out_full = pipe(prompt=PROMPT, num_inference_steps=8, guidance_scale=0.0, width=512, height=512,
                generator=g, output_type="latent", callback_on_step_end=cb)
print(f"\n[Q-baseline] num_inference_steps=8 → callback 触发 {len(seen)} 次; 末步 sigma={seen[-1][2] if seen else None}")
lat_full = out_full.images if not isinstance(out_full.images,(list,tuple)) else out_full.images[0]
print(f"  full latent shape={tuple(lat_full.shape)} mean={lat_full.float().mean():.4f} std={lat_full.float().std():.4f}")

# Q1b: 传 custom sigmas 只到中段(不含 0)——看是否 append 0 / 停在哪
import inspect
sig_param = "sigmas" in inspect.signature(pipe.__call__).parameters
print(f"\n[Q1] __call__ 接受 sigmas=? {sig_param}")
if sig_param:
    # 取 baseline 8 步的 sigma 表,截前 5 个(不到 0)
    base_sigmas = None
    try:
        from src.services.inference.sigma_schedules import compute_sigmas  # noqa
    except Exception:
        import sys
        sys.path.insert(0, ".")
        from src.services.inference.sigma_schedules import compute_sigmas
    full = compute_sigmas("simple", 8, shift=float(getattr(pipe.scheduler.config,"shift",3.0) or 3.0))
    print(f"  compute_sigmas('simple',8)={[round(x,3) for x in full]}")
    partial = full[:6]  # [s0..s5],不含 0 → 期望 5 步,停在 s5(leftover)
    seen.clear()
    g = torch.Generator(DEVICE).manual_seed(42)
    try:
        out_p = pipe(prompt=PROMPT, sigmas=partial[:-1] if partial[-1]==0 else partial, guidance_scale=0.0,
                     width=512, height=512, generator=g, output_type="latent", callback_on_step_end=cb)
        lat_p = out_p.images if not isinstance(out_p.images,(list,tuple)) else out_p.images[0]
        print(f"  传 sigmas={[round(x,3) for x in partial]} → callback {len(seen)} 次; "
              f"末步 sigma={seen[-1][2] if seen else None}; latent std={lat_p.float().std():.4f}")
        print(f"  各步 sigma: {[round(s[2],3) if s[2] else s[2] for s in seen]}")
    except Exception as e:
        print(f"  传 partial sigmas 报错: {type(e).__name__}: {e}")

# Q2: 注入 latents= + 短 sigmas，看是否原样续采
print("\n[Q2] latents= 注入 + 中段 sigmas")
g = torch.Generator(DEVICE).manual_seed(7)
shape = lat_full.shape
inj = torch.randn(shape, generator=g, device=DEVICE, dtype=lat_full.dtype)
seen.clear()
g2 = torch.Generator(DEVICE).manual_seed(99)
try:
    full = compute_sigmas("simple", 8, shift=3.0)
    tail = full[5:]  # [s5..0]
    out_inj = pipe(prompt=PROMPT, sigmas=tail[:-1] if tail[-1]==0 else tail, guidance_scale=0.0,
                   width=512, height=512, generator=g2, latents=inj.clone(), output_type="latent",
                   callback_on_step_end=cb)
    lat_inj = out_inj.images if not isinstance(out_inj.images,(list,tuple)) else out_inj.images[0]
    # 第 0 步前 latent 应≈inj(若原样);比第一次 callback 的 latent 与 inj 差异
    first = seen[0][3] if seen else None
    print(f"  注入 latents std={inj.float().std():.4f}; 跑 {len(seen)} 步; 末 latent std={lat_inj.float().std():.4f}")
    print(f"  各步 sigma: {[round(s[2],3) if s[2] else s[2] for s in seen]}")
except Exception as e:
    print(f"  latents= 注入报错: {type(e).__name__}: {e}")

print("\n探针完成。")
