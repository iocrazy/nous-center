"""可行性 spike:Ideogram-4 双 DiT 单文件 **offload 机制**(TE+VAE 驻卡 + 双 DiT cpu stash 轮转)。需 GPU。

策略(2026-06-13 用户拍):**只卸载两个 DiT**(大件),TE+VAE 驻卡。关键 —— TE 不卸载 → diffusers
pipeline 直调 `text_encoder.language_model.embed_tokens(token_ids)` 时 embed_tokens 权重始终在卡上
→ **绕开 spike_ideogram4_singlefile 逮到的 cpu-offload device 错配**。

验:① TE 驻卡时不再有 embed_tokens device 错配;② 双 DiT cpu-stash 轮转(一个时刻只一个 DiT 上卡)
→ 峰值从「全驻 53G」降到「TE+1DiT ~35G」(bf16);③ 出连贯图。
→ 验通即证「逐组件 offload 只卸 DiT」可行,实施改 _place_components_per_device 接 unconditional_transformer。

bf16 机制验证用 cuda:1(97G,峰值 ~35G);fp8 塞 24G 3090 是实施阶段的真验证(需 torchao 量化)。
用法:cd backend && SMOKE_DEVICE=cuda:1 SMOKE_SIZE=512 uv run python tests/manual/spike_ideogram4_offload.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PKG = "/media/heygo/Program/models/【83】Ideogram4全自动流程(1)/models"
DIT = f"{PKG}/diffusion_models/ideogram4_fp8_scaled.safetensors"
DIT_U = f"{PKG}/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
TE = f"{PKG}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
VAE = f"{PKG}/vae/flux2-vae.safetensors"
REPO = os.environ.get("SMOKE_IDEO_FULL", "/media/heygo/Program/models/nous/image/diffusers/Ideogram-4-bf16")
DEV = os.environ.get("SMOKE_DEVICE", "cuda:1")
SIZE = int(os.environ.get("SMOKE_SIZE", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "12"))
PROMPT = "a photo of a red fox in autumn leaves, sharp focus"
SEED = 42


def _wrap_dit_offload(module, compute_dev):
    """把 DiT 留 CPU,forward 前移到 compute 卡、forward 后挪回 CPU(最小 cpu-stash 轮转)。
    复刻产品 _place_components_per_device 对 transformer 的逐组件 offload,但手动包,验机制。"""
    import torch  # noqa: PLC0415
    orig = module.forward

    def wrapped(*args, **kwargs):
        module.to(compute_dev)
        moved_a = [a.to(compute_dev) if torch.is_tensor(a) else a for a in args]
        moved_k = {k: (v.to(compute_dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        out = orig(*moved_a, **moved_k)
        module.to("cpu")
        torch.cuda.empty_cache()
        return out
    module.forward = wrapped


async def main() -> int:
    import torch  # noqa: PLC0415
    from diffusers import FlowMatchEulerDiscreteScheduler, Ideogram4Pipeline  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from src.services.inference.component_spec import ComponentSpec  # noqa: PLC0415
    from src.services.inference.image_modular import (  # noqa: PLC0415
        build_bridged_text_encoder,
        build_bridged_transformer,
        build_bridged_vae,
    )
    torch.cuda.reset_peak_memory_stats(torch.device(DEV))

    # DiT 建在 CPU(随后 cpu-stash 轮转);TE/VAE 建在卡上(驻卡)。
    print(f"① 建 4 组件:双 DiT→CPU(stash),TE/VAE→{DEV}(驻卡)… size={SIZE}")
    t = build_bridged_transformer(ComponentSpec(kind="diffusion_models", file=DIT, device="cpu",
                                  dtype="bfloat16", adapter_arch="ideogram4"), REPO, "cpu", "transformer")
    tu = build_bridged_transformer(ComponentSpec(kind="diffusion_models", file=DIT_U, device="cpu",
                                   dtype="bfloat16", adapter_arch="ideogram4"), REPO, "cpu", "unconditional_transformer")
    te = build_bridged_text_encoder(ComponentSpec(kind="clip", file=TE, device=DEV, dtype="bfloat16"), REPO, DEV)
    vae = build_bridged_vae(ComponentSpec(kind="vae", file=VAE, device=DEV, dtype="bfloat16"), REPO, DEV)
    _wrap_dit_offload(t, DEV)
    _wrap_dit_offload(tu, DEV)
    print("   双 DiT cpu-stash 轮转 forward 包好;TE/VAE 驻卡")

    tok = AutoTokenizer.from_pretrained(str(Path(REPO) / "tokenizer"))
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(str(Path(REPO) / "scheduler"))
    pipe = Ideogram4Pipeline(scheduler=sched, vae=vae, text_encoder=te, tokenizer=tok,
                             transformer=t, unconditional_transformer=tu)
    print("② 出图(双 DiT 轮转 cpu↔gpu;TE 驻卡 → 无 embed_tokens device 错配)…")
    n_low = max(1, STEPS // 4)
    gsched = [7.0] * (STEPS - n_low) + [3.0] * n_low
    g = torch.Generator(device="cpu").manual_seed(SEED)
    img = pipe(prompt=PROMPT, width=SIZE, height=SIZE, num_inference_steps=STEPS,
               guidance_schedule=gsched, generator=g).images[0]
    peak = torch.cuda.max_memory_allocated(torch.device(DEV)) / 1024**2
    p = Path(tempfile.gettempdir()) / "ideo_offload.png"
    img.save(p)
    coherent = p.stat().st_size > 10000
    # 峰值判据:全驻 bf16 ~53G;DiT 轮转后应 < 42G(TE 16 + 1 DiT 18.6 + VAE + 激活)。
    peak_ok = peak < 42000
    print(f"   出图 → {p} ({p.stat().st_size // 1024}KB) peak_vram={peak:.0f}MB "
          f"{'✓连贯' if coherent else 'FAIL'} {'✓峰值降(<42G,DiT 轮转生效)' if peak_ok else 'FAIL 峰值未降'}")
    ok = coherent and peak_ok
    print(f"\n{'✅ spike PASS:TE 驻卡绕开 embed_tokens 错配 + 双 DiT cpu-stash 降峰值 → 机制可行' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
