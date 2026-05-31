"""PR-anima-1 真模型 forward smoke(CPU 即可,无 GPU 也跑)。

验 Cosmos Predict2 MiniTrainDIT(及其子组件)在真 torch 下能构造 + forward 跑通 +
输出形状/数值合理。不验跟 anima 权重的 SSIM —— 那是 PR-anima-7(spec 2026-05-26-anima-port-design)。

用法:
    cd backend
    uv run python tests/manual/smoke_anima_pr1.py

CPU 跑约 5-10s(小配置,1 block)。CI 跑不了真 forward(conftest mock torch),所以这是
唯一的「真 nn.Module 在 PR-anima-1 层级」回归门(feedback_verify_real_model)。
"""
from __future__ import annotations


import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # standalone:cuda:1=Pro 6000,对齐 nvidia-smi(torch 默认 FASTEST_FIRST 会翻卡)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import (  # noqa: PLC0415
        Attention,
        MiniTrainDIT,
        PatchEmbed,
        Timesteps,
    )

    print("[anima-pr1] testing imports + small MiniTrainDIT forward")

    # —— 子组件 sanity check ——
    self_attn = Attention(query_dim=64, n_heads=4, head_dim=16)
    x = torch.randn(2, 8, 64)
    assert self_attn(x).shape == (2, 8, 64)
    print("  ✓ Attention(self):", self_attn(x).shape)

    cross_attn = Attention(query_dim=64, context_dim=32, n_heads=4, head_dim=16)
    ctx = torch.randn(2, 5, 32)
    assert cross_attn(x, ctx).shape == (2, 8, 64)
    print("  ✓ Attention(cross):", cross_attn(x, ctx).shape)

    pe = PatchEmbed(spatial_patch_size=2, temporal_patch_size=1, in_channels=4, out_channels=96)
    pe_out = pe(torch.randn(1, 4, 1, 16, 16))
    assert pe_out.shape == (1, 1, 8, 8, 96)
    print("  ✓ PatchEmbed:", pe_out.shape)

    ts = Timesteps(num_channels=128)
    ts_out = ts(torch.tensor([[0.1, 0.5], [0.9, 0.3]]))
    assert ts_out.shape == (2, 2, 128)
    print("  ✓ Timesteps:", ts_out.shape)

    # —— 主类 MiniTrainDIT 小配置 forward ——
    # Anima 真配置(估):max_img=128,patch=2,channels=128。这里用最小可跑配置。
    model = MiniTrainDIT(
        max_img_h=32, max_img_w=32, max_frames=2,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=96, num_blocks=2, num_heads=6, mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
        use_adaln_lora=False,
        rope_enable_fps_modulation=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ MiniTrainDIT built ({n_params/1e6:.2f}M params, {model.num_blocks} blocks)")

    x = torch.randn(1, 4, 1, 16, 16)
    timesteps = torch.tensor([0.5])  # (B,)
    context = torch.randn(1, 5, 128)  # (B, N=5, D=128)

    with torch.no_grad():
        out = model(x, timesteps, context)

    assert out.shape == (1, 4, 1, 16, 16), f"unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "output contains nan/inf"
    print(f"  ✓ MiniTrainDIT forward: {out.shape}, finite={torch.isfinite(out).all().item()}")
    print(f"  ✓ output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    # —— use_adaln_lora 也跑通(Anima 用这个) ——
    model_lora = MiniTrainDIT(
        max_img_h=32, max_img_w=32, max_frames=2,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=96, num_blocks=1, num_heads=6,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
        use_adaln_lora=True, adaln_lora_dim=32,
        rope_enable_fps_modulation=False,
    )
    with torch.no_grad():
        out_lora = model_lora(x, timesteps, context)
    assert out_lora.shape == (1, 4, 1, 16, 16)
    print(f"  ✓ MiniTrainDIT(use_adaln_lora=True) forward: {out_lora.shape}")

    print("[anima-pr1] verdict = PASS")


if __name__ == "__main__":
    main()
