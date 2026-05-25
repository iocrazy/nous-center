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


# NVFP4 E2M1 码 → 幅值(标准 E2M1:1符号+2指数+1尾数;code&7 取幅,bit3 符号)
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _dequant_comfy_mixed(file: str):
    """逐张量 comfy 混合量化反量化(.comfy_quant JSON 驱动:fp8 / nvfp4 / plain)→ bf16 state_dict。

    PR-1 spike 验:qwen 文件 = float8_e4m3fn×141 + nvfp4×85 + bf16×172。
    nvfp4 dequant = E2M1解码(uint8 unpack)× block_scale(fp8 [out,in/16]) × global_scale(fp32 标量)。
    """
    import json

    import torch
    from safetensors.torch import load_file

    raw = load_file(file)
    e2m1 = torch.tensor(_E2M1_MAG, dtype=torch.float32)

    def _qfmt(base):
        cq = raw.get(base + ".comfy_quant")
        if cq is None:
            return None
        return json.loads(bytes(cq.tolist()).decode("utf-8")).get("format")

    clean: dict[str, torch.Tensor] = {}
    n_fp8 = n_nvfp4 = n_plain = 0
    for key, t in raw.items():
        if key.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2")):
            continue
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        fmt = _qfmt(base) if base else None
        if fmt == "float8_e4m3fn":
            scale = raw[base + ".weight_scale"].to(torch.float32)
            clean[key] = (t.to(torch.float32) * scale).to(torch.bfloat16)
            n_fp8 += 1
        elif fmt == "nvfp4":
            bs = raw[base + ".weight_scale"]          # fp8 [out, in/16]
            gs = raw[base + ".weight_scale_2"].to(torch.float32)  # fp32 标量
            out = t.shape[0]
            low = (t & 0x0F).to(torch.long)
            high = ((t >> 4) & 0x0F).to(torch.long)
            dec = lambda c: torch.where(c >= 8, -1.0, 1.0) * e2m1[c & 0x7]  # noqa: E731
            # comfy 打包:packed = (even<<4) | odd → 偶数元素=高 nibble、奇数元素=低 nibble
            vals = torch.stack([dec(high), dec(low)], dim=-1).reshape(out, -1)  # [out, in] high(even)-then-low(odd)
            blk = bs.to(torch.float32).repeat_interleave(16, dim=1)            # [out, in]
            clean[key] = (vals * blk * gs).to(torch.bfloat16)
            n_nvfp4 += 1
        else:
            clean[key] = t.to(torch.bfloat16) if t.dtype in (torch.float32, torch.float16, torch.bfloat16) else t
            n_plain += 1
    print(f"[single-file] comfy-mixed dequant: fp8={n_fp8} nvfp4={n_nvfp4} plain={n_plain}")
    return clean


def build_bridged_text_encoder(clip_file: str, repo: str, device: str, dtype: str = "bfloat16"):
    """单文件 text encoder → repo config 建 Qwen3 + comfy 逐张量混合反量化(标准键,无需转)。"""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3ForCausalLM

    sd = _dequant_comfy_mixed(clip_file)
    cfg = AutoConfig.from_pretrained(f"{repo}/text_encoder")
    with init_empty_weights():
        model = Qwen3ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[single-file] text_encoder missing={len(missing)}({missing[:3]}) unexpected={len(unexpected)}")
    model.tie_weights()  # lm_head 与 embed_tokens 绑定(comfy 文件省了 tied lm_head)→ 实体化
    # 仍残留 meta 的 param(Flux2 文本编码不用,如未绑的 lm_head)→ 逐个零初始化(只动 meta,不碰已加载)
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        print(f"[single-file] 仍 meta(零初始化兜底): {meta[:3]} (#{len(meta)})")
        for name in meta:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[1]
            p = getattr(parent, attr)
            setattr(parent, attr, torch.nn.Parameter(
                torch.zeros(p.shape, dtype=torch.bfloat16), requires_grad=False))
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
    # 隔离开关:SPIKE_TE=repo → 用 repo bf16 text encoder(不 override),验 NVFP4 是否唯一锅。
    overrides = {"transformer": transformer, "vae": vae}
    if os.environ.get("SPIKE_TE") != "repo":
        overrides["text_encoder"] = text_encoder
    else:
        print("[single-file] 隔离:用 repo bf16 text_encoder(跳过 NVFP4 单文件)")
    pipe.update_components(**overrides)
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
