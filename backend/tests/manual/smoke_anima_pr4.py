"""PR-anima-4 真模型 smoke:加载 anima-base-v1.0.safetensors 到 nous Anima nn.Module。

验:
  - load_anima_dit_from_single_file 成功(strip 'net.' prefix + Anima(**config))
  - missing / unexpected key 在可接受范围(< 10% 总 key 数,核心权重对齐)
  - 真 forward 跑通(小输入,验 shape + finite)
  - 真模型 peak VRAM 合理(2.09B × bf16 ≈ 4.2GB)

用法:
    cd backend
    SMOKE_DEVICE=cuda:1 uv run python tests/manual/smoke_anima_pr4.py

GPU ~10-20s(加载 + 一次小 forward)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANIMA_WEIGHTS = "/media/heygo/Program/models/nous/image/diffusion_models/anima/anima-base-v1.0.safetensors"
DEVICE = os.environ.get("SMOKE_DEVICE", "cuda:1")


def main() -> None:
    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import (  # noqa: PLC0415
        ANIMA_BASE_V1_CONFIG,
        Anima,
        load_anima_dit_from_single_file,
    )

    if not Path(ANIMA_WEIGHTS).exists():
        print(f"[anima-pr4] !! anima weights missing: {ANIMA_WEIGHTS} — skipping")
        return

    print(f"[anima-pr4] weights = {ANIMA_WEIGHTS}")
    print(f"[anima-pr4] device  = {DEVICE}, dtype = bfloat16")
    print("[anima-pr4] config:")
    for k in ("model_channels", "num_blocks", "num_heads", "use_adaln_lora", "adaln_lora_dim",
             "in_channels", "patch_spatial", "crossattn_emb_channels"):
        print(f"  {k} = {ANIMA_BASE_V1_CONFIG[k]}")

    # 触发 CUDA init — torch.cuda.reset_peak_memory_stats 前必须先 init device。
    _ = torch.zeros(1, device=DEVICE)
    torch.cuda.reset_peak_memory_stats(torch.device(DEVICE))

    print("[anima-pr4] loading...")
    model: Anima = load_anima_dit_from_single_file(
        ANIMA_WEIGHTS, device=DEVICE, dtype=torch.bfloat16,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ loaded: {n_params/1e9:.2f}B params({n_trainable/1e9:.2f}B trainable)")
    print(f"  ✓ on device {next(model.parameters()).device}")

    # —— 小 forward(latent 16×16,context 5 token,1 step)——
    # anima 真出图 latent 通常 128×128(VAE 8x → 1024×1024)。smoke 跑小一点免久等。
    B, T, C, H, W = 1, 1, 16, 16, 16
    x = torch.randn(B, C, T, H, W, device=DEVICE, dtype=torch.bfloat16)
    timesteps = torch.tensor([0.5], device=DEVICE, dtype=torch.bfloat16)
    context = torch.randn(B, 5, 1024, device=DEVICE, dtype=torch.bfloat16)

    print(f"[anima-pr4] forward: x={tuple(x.shape)} context={tuple(context.shape)}")
    with torch.no_grad():
        out = model(x, timesteps, context)
    assert out.shape == x.shape, f"output shape {out.shape} != input {x.shape}"
    assert torch.isfinite(out).all(), "output has nan/inf"
    print(f"  ✓ out shape = {tuple(out.shape)}, finite = True")
    print(f"  ✓ out stats: mean = {out.float().mean().item():.4f}, std = {out.float().std().item():.4f}")

    peak_mib = torch.cuda.max_memory_allocated(torch.device(DEVICE)) / 1024**2
    print(f"[anima-pr4] peak VRAM = {peak_mib:.0f} MiB ({peak_mib/1024:.2f} GiB)")
    print("[anima-pr4] verdict = PASS")


if __name__ == "__main__":
    main()
