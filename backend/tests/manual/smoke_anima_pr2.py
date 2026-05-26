"""PR-anima-2 真模型 forward smoke(CPU 即可)。

验 Anima(继承 MiniTrainDIT)+ LLMAdapter + 1D RoPE TransformerBlock 真 forward 跑通。
qwen3 标准路径 +(可选)t5xxl 桥接路径都要验。

用法:
    cd backend
    uv run python tests/manual/smoke_anima_pr2.py

CPU 跑约 10-20s。spec 真模型 SSIM 验留 PR-anima-7(对照 anima ComfyUI 工作流)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    import torch  # noqa: PLC0415

    from src.services.inference.arch_anima import (  # noqa: PLC0415
        Anima,
        LLMAdapter,
        RotaryEmbedding,
    )

    print("[anima-pr2] testing LLMAdapter + Anima forward")

    # —— 1D RoPE sanity ——
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(1, 8, 4, 64)  # (B, S, H, D) 标准 LLM shape
    pos_ids = torch.arange(8).unsqueeze(0)  # (B=1, S=8)
    cos, sin = rope(x, pos_ids)
    assert cos.shape == (1, 8, 64) and sin.shape == (1, 8, 64), (
        f"unexpected rope shapes: cos={cos.shape} sin={sin.shape}"
    )
    print(f"  ✓ RotaryEmbedding: cos {cos.shape}, sin {sin.shape}")

    # —— LLMAdapter 独立 forward ——
    # source = qwen embeds(B=1, L_src=10, D=1024);target_ids = t5xxl tokens(B=1, L_t=12)
    adapter = LLMAdapter(
        source_dim=1024, target_dim=1024, model_dim=1024,
        num_layers=2,  # smoke 用小层数;真模型 6 层
        num_heads=16,
    )
    source = torch.randn(1, 10, 1024)
    target_ids = torch.randint(0, 32128, (1, 12))
    with torch.no_grad():
        out = adapter(source, target_ids)
    assert out.shape == (1, 12, 1024), f"LLMAdapter shape {out.shape} != (1,12,1024)"
    assert torch.isfinite(out).all()
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  ✓ LLMAdapter forward: {out.shape}  ({n_params/1e6:.1f}M params, {len(adapter.blocks)} blocks)")

    # —— Anima 全模型(继承 MiniTrainDIT)forward,**无 t5xxl** ——
    # 用 PR-anima-1 smoke 同样的小配置。
    model = Anima(
        max_img_h=32, max_img_w=32, max_frames=2,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=96, num_blocks=1, num_heads=6, mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
        use_adaln_lora=False,
        rope_enable_fps_modulation=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Anima built ({n_params/1e6:.1f}M params; LLMAdapter 6 层 ~16M / DiT 主干 ~0.4M)")

    x = torch.randn(1, 4, 1, 16, 16)
    timesteps = torch.tensor([0.5])
    context = torch.randn(1, 5, 128)  # qwen embeds(N=5 tokens)

    with torch.no_grad():
        out_no_t5 = model(x, timesteps, context)
    assert out_no_t5.shape == (1, 4, 1, 16, 16)
    print(f"  ✓ Anima forward (no t5xxl, qwen 路径): {out_no_t5.shape}")

    # —— Anima 走 t5xxl 桥接路径 ——
    # llm_adapter 默认 source_dim=target_dim=model_dim=1024,但我们 crossattn_emb=128 不匹配。
    # 重建一个 source=128 的 adapter 验路径(真模型 LLMAdapter 用默认 dim,这里只验 forward 不挂)。
    # 实际 Anima 主类 LLMAdapter 是默认 1024 dim;t5xxl context 也必须 1024 dim 才能匹配。
    # 这里用同 dim 配置专门测 t5xxl 路径。
    model_t5 = Anima(
        max_img_h=32, max_img_w=32, max_frames=2,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=96, num_blocks=1, num_heads=6, mlp_ratio=2.0,
        crossattn_emb_channels=1024,  # 跟 LLMAdapter target_dim 对齐
        pos_emb_cls="rope3d",
        use_adaln_lora=False,
        rope_enable_fps_modulation=False,
    )
    context_qwen = torch.randn(1, 5, 1024)  # source = qwen embeds(D=1024)
    t5_ids = torch.randint(0, 32128, (1, 8))  # target = t5xxl token ids(L_t=8)
    with torch.no_grad():
        out_with_t5 = model_t5(x, timesteps, context_qwen, t5xxl_ids=t5_ids)
    assert out_with_t5.shape == (1, 4, 1, 16, 16)
    assert torch.isfinite(out_with_t5).all()
    print(f"  ✓ Anima forward (with t5xxl_ids, 桥接路径): {out_with_t5.shape}")

    print("[anima-pr2] verdict = PASS")


if __name__ == "__main__":
    main()
