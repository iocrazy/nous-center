"""PR-1 Task 1.0 — 量化文件紧凑加载方案 spike(torchao fp8)。standalone,需 GPU。

验方案 A(diffusers/torchao Float8WeightOnlyConfig):量化 transformer + text_encoder 后
  - model.dtype 仍报 bf16(绕开 PR-0 spike 撞的 modular noise-dtype bug);
  - 权重 fp8 紧凑驻留(~½ 显存);
  - 完整 modular pipe 在 3090 出**正确狐狸图** + 进 24GB。

对照 PR-0 的 layerwise-casting(model.dtype 变 fp8 → randn 崩)。torchao 微验证已确认
transformer 量化后 .dtype=bf16 + 17GB→8.7GB。本 spike 验完整出图。

用法:
    cd backend
    SPIKE_DEVICE=cuda:2 uv run python tests/manual/spike_quant_compact.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:0/2=3090, cuda:1=Pro6000

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = "/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"
DEVICE = os.environ.get("SPIKE_DEVICE", "cuda:2")  # 约束卡 3090 24GB
PROMPT = "a photo of a red fox sitting in autumn leaves, sharp focus, detailed"
OUT_DIR = Path(__file__).parent / "_smoke_out"


def main() -> int:
    import torch
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    from src.services.inference.image_modular import _import_modular, _torch_dtype

    dev = torch.device(DEVICE)
    props = torch.cuda.get_device_properties(dev)
    print(f"[quant-compact] device={DEVICE} ({props.name} {props.total_memory/1024**3:.1f}GB)")
    torch.cuda.reset_peak_memory_stats(dev)

    modular_pipeline_cls, components_manager_cls = _import_modular()
    pipe = modular_pipeline_cls.from_pretrained(REPO, components_manager=components_manager_cls())
    pipe.load_components(torch_dtype=_torch_dtype("bfloat16"))

    t_q = time.monotonic()
    cfg = Float8WeightOnlyConfig()
    for name in ("transformer", "text_encoder"):
        mod = getattr(pipe, name, None)
        if mod is None:
            print(f"[quant-compact] {name} 缺失,跳过")
            continue
        quantize_(mod, cfg)
        print(f"[quant-compact] quantized {name} fp8-wo; .dtype={getattr(mod, 'dtype', '?')}")
    print(f"[quant-compact] quantize in {time.monotonic()-t_q:.1f}s")

    pipe.to(dev)
    mem_load = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(f"[quant-compact] peak-after-load={mem_load:.0f}MB")

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    t_inf = time.monotonic()
    try:
        out = pipe(prompt=PROMPT, num_inference_steps=20, width=1024, height=1024, generator=gen)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[quant-compact] OOM during inference: {e}")
        return 0
    inf_ms = int((time.monotonic() - t_inf) * 1000)
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2

    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / "spike_quant_compact.png"
    out_png.write_bytes(buf.getvalue())
    print(f"[quant-compact] saved → {out_png} ({len(buf.getvalue())} bytes)")
    print(f"[quant-compact] RESULT peak_vram={peak:.0f}MB infer_ms={inf_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
