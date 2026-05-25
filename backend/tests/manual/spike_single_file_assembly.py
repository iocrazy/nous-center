"""PR-1 spike — 单文件流水线装配(ComfyUI 式)。standalone,需 GPU。

验:transformer + text_encoder + vae **全用单文件**,config/scheduler/tokenizer 取自「架构参考整模型」
(diffusers/Flux2-klein-9B),三组件都 override 进 ModularPipeline → 出正确图。

桥接:
  - transformer: 现有 build_bridged_transformer(dequant+flux2 转键 + repo config)。
  - text_encoder(新): QUANT_LOADERS.dispatch(只反量化,qwen 是标准 Qwen3 键无需转)→ 注入 repo config 建的 Qwen3。
  - vae(新): dispatch(plain)→ 注入 repo config 建的 AutoencoderKLFlux2。

用法:
    cd backend
    SPIKE_DEVICE=cuda:1 uv run python tests/manual/spike_single_file_assembly.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:1=Pro6000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMG = "/media/heygo/Program/models/nous/image"
REPO = f"{IMG}/diffusers/Flux2-klein-9B"          # 架构参考整模型(config/scheduler/tokenizer)
UNET = f"{IMG}/diffusion_models/flux/Flux2-Klein-9B-True-v2-bf16.safetensors"  # 单文件 transformer
CLIP = f"{IMG}/text_encoders/qwen_3_8b_fp8mixed.safetensors"                   # 单文件 clip(comfy fp8)
VAE = f"{IMG}/vae/flux2-vae.safetensors"                                       # 单文件 vae
DEVICE = os.environ.get("SPIKE_DEVICE", "cuda:1")
OUT_DIR = Path(__file__).parent / "_smoke_out"


def build_bridged_text_encoder(clip_file: str, repo: str, device: str, dtype: str = "bfloat16"):
    """单文件 text encoder → repo config 建 Qwen3 + 反量化单文件权重(标准键,无需转)。"""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3ForCausalLM

    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.quant_loaders import QUANT_LOADERS

    sd = QUANT_LOADERS.dispatch(ComponentSpec(kind="clip", file=clip_file, device=device, dtype=dtype))
    cfg = AutoConfig.from_pretrained(f"{repo}/text_encoder")
    with init_empty_weights():
        model = Qwen3ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[single-file] text_encoder missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device, dtype=torch.bfloat16)


def build_bridged_vae(vae_file: str, repo: str, device: str, dtype: str = "bfloat16"):
    """单文件 vae → repo config 建 AutoencoderKLFlux2 + 单文件权重(plain,标准键)。"""
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2

    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.quant_loaders import QUANT_LOADERS

    sd = QUANT_LOADERS.dispatch(ComponentSpec(kind="vae", file=vae_file, device=device, dtype=dtype))
    cfg = AutoencoderKLFlux2.load_config(f"{repo}/vae")
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(cfg)
    missing, unexpected = vae.load_state_dict(sd, strict=False, assign=True)
    print(f"[single-file] vae missing={len(missing)} unexpected={len(unexpected)}")
    return vae.to(device, dtype=torch.bfloat16)


def main() -> int:
    import torch

    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.image_modular import _import_modular, build_bridged_transformer

    dev = torch.device(DEVICE)
    props = torch.cuda.get_device_properties(dev)
    print(f"[single-file] device={DEVICE} ({props.name} {props.total_memory/1024**3:.1f}GB)")
    torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.monotonic()
    transformer = build_bridged_transformer(
        ComponentSpec(kind="diffusion_models", file=UNET, device=DEVICE, dtype="bfloat16", adapter_arch="flux2"),
        REPO, DEVICE)
    print(f"[single-file] transformer built ({time.monotonic()-t0:.1f}s)")
    text_encoder = build_bridged_text_encoder(CLIP, REPO, DEVICE)
    vae = build_bridged_vae(VAE, REPO, DEVICE)
    print(f"[single-file] all 3 single-file components built ({time.monotonic()-t0:.1f}s)")

    modular_pipeline_cls, components_manager_cls = _import_modular()
    pipe = modular_pipeline_cls.from_pretrained(REPO, components_manager=components_manager_cls())
    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.update_components(transformer=transformer, text_encoder=text_encoder, vae=vae)
    pipe.to(dev)

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    t_inf = time.monotonic()
    out = pipe(prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed",
               num_inference_steps=20, width=1024, height=1024, generator=gen)
    inf_ms = int((time.monotonic() - t_inf) * 1000)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2

    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / "spike_single_file_assembly.png"
    out_png.write_bytes(buf.getvalue())
    print(f"[single-file] saved → {out_png} ({len(buf.getvalue())} bytes)")
    print(f"[single-file] RESULT peak_vram={peak:.0f}MB infer_ms={inf_ms} — 三单文件装配出图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
