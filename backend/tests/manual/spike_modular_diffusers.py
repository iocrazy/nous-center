"""SPIKE: Modular Diffusers 可行性验证(落 Pro 6000 / cuda:1)。

目的:用数据决定是否用 Modular Diffusers(ModularPipeline + ComponentsManager)
替换自写 ImageSampler / 组件装配。三阶段,全 cuda:1(Pro 6000 96GB,不 offload):

  A 基线:ModularPipeline.from_pretrained(HF-layout Flux2-klein)+ load_components(bf16)
          → 出图(seed 42 / 20 步 / 1024²,与现有 smoke_cuda1_default.png 同条件可对比)
  B GGUF:Flux2Transformer2DModel.from_single_file(Q5_K.gguf, GGUFQuantizationConfig)
          → update_components → 出图(当前自建栈 reject_gguf,跑不了)
  C 单文件 fp8mixed:from_single_file(...-fp8mixed.safetensors)→ update_components → 出图

每阶段 try/except 独立,存图 + 打印耗时/错误。跑:
  CUDA_DEVICE_ORDER=PCI_BUS_ID uv run python tests/manual/spike_modular_diffusers.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

REPO = "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"
FLUX_DIR = "/media/heygo/Program/models/nous/image/diffusion_models/flux"
GGUF = f"{FLUX_DIR}/Flux2-Klein-9B-True-v2-Q5_K.gguf"
FP8 = f"{FLUX_DIR}/Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
DEVICE = "cuda:1"
OUT = Path(__file__).parent / "_smoke_out"
OUT.mkdir(exist_ok=True)
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
GEN_KW = dict(num_inference_steps=20, height=1024, width=1024)


def _gen(pipe, tag):
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    t = time.monotonic()
    out = pipe(prompt=PROMPT, generator=gen, **GEN_KW)
    img = out.images[0]
    dt = time.monotonic() - t
    p = OUT / f"spike_{tag}.png"
    img.save(p)
    print(f"[spike:{tag}] infer {dt:.1f}s → {p}")
    return dt


def stage_a_baseline():
    print("\n===== Stage A: ModularPipeline 基线(HF-layout bf16, cuda:1) =====")
    from diffusers import ModularPipeline
    t = time.monotonic()
    pipe = ModularPipeline.from_pretrained(REPO)
    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.to(DEVICE)
    print(f"[spike:A] load {time.monotonic()-t:.1f}s; blocks={type(pipe).__name__}")
    _gen(pipe, "A_modular_bf16")
    return pipe


def stage_b_gguf(pipe):
    print("\n===== Stage B: GGUF 单文件 transformer(当前栈跑不了) =====")
    from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
    t = time.monotonic()
    tr = Flux2Transformer2DModel.from_single_file(
        GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=f"{REPO}/transformer",
    ).to(DEVICE)
    print(f"[spike:B] GGUF transformer load {time.monotonic()-t:.1f}s")
    pipe.update_components(transformer=tr)
    _gen(pipe, "B_gguf_q5k")


def stage_c_fp8(pipe):
    print("\n===== Stage C: 单文件 fp8mixed transformer =====")
    from diffusers import Flux2Transformer2DModel
    t = time.monotonic()
    tr = Flux2Transformer2DModel.from_single_file(
        FP8, torch_dtype=torch.bfloat16, config=f"{REPO}/transformer",
    ).to(DEVICE)
    print(f"[spike:C] fp8mixed transformer load {time.monotonic()-t:.1f}s")
    pipe.update_components(transformer=tr)
    _gen(pipe, "C_fp8mixed")


def main():
    pipe = None
    for name, fn in [("A", stage_a_baseline)]:
        try:
            pipe = fn()
        except Exception:
            print(f"[spike:{name}] FAILED:")
            traceback.print_exc()
    for name, fn in [("B", stage_b_gguf), ("C", stage_c_fp8)]:
        if pipe is None:
            print(f"[spike:{name}] 跳过(基线 pipe 未建成)")
            continue
        try:
            fn(pipe)
        except Exception:
            print(f"[spike:{name}] FAILED:")
            traceback.print_exc()
    print("\n[spike] done")


if __name__ == "__main__":
    main()
