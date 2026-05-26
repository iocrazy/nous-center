"""Anima 单文件权重加载器 — 从 anima-base-v1.0.safetensors 等 build Anima nn.Module。

接 PR-anima-1/2/3。spec 2026-05-26-anima-port-design **PR-4 决策**:
  anima-base-v1.0.safetensors 全 bf16 无量化 → 标准 `nn.Linear` 够,
  comfy.operations 简化(spec PR-4)**跳过** —— 真有量化版 anima 出来再做。

## 关键细节

权重 key 命名:全部以 `net.*` 前缀(685/685 keys);nous Anima nn.Module 顶层直接是
组件名(`blocks` / `t_embedder` / `llm_adapter` 等)→ load 时 strip 'net.' 即对齐。

## Anima-base-v1.0 推断 config(从权重 shape 反推)

| 字段 | 值 | 推断来源 |
|---|---|---|
| in_channels | 16 | x_embedder.proj.1.weight (2048, 68) → (16+1)·1·2·2 = 68 |
| out_channels | 16 | final_layer.linear.weight (64, 2048) → 2·2·1·16 = 64 |
| patch_spatial | 2 | 同上 |
| patch_temporal | 1 | image 模型,T=1 |
| model_channels | 2048 | t_embedder.1.linear_1.weight (2048, 2048) |
| num_blocks | 28 | net.blocks.0..27 |
| num_heads | 16 | cross_attn.k_norm.weight (128,) → head_dim=128 → 2048/128 |
| crossattn_emb_channels | 1024 | cross_attn.k_proj.weight (2048, 1024) |
| use_adaln_lora | True | t_embedder.1.linear_2.weight (6144, 2048) ≠ (2048, 2048) |
| adaln_lora_dim | 256 | adaln_modulation_*.1.weight (256, 2048) |
| LLMAdapter num_layers | 6 | net.llm_adapter.blocks.0..5 |
| concat_padding_mask | True | x_embedder in 68 = (16+1)·4 |
| pos_emb_cls | "rope3d" | (按 spec,无对应权重 — register_buffer 派生) |
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


# anima-base-v1.0 全模型固定配置(从权重 shape 推断)。
# max_img_h/w/max_frames 是 pos_embedder buffer 派生(不在 state_dict 里),按 anima 1024×1024 输出
# + VAE 8× downsample → latent 128×128 → patch_spatial=2 → grid 64×64 取保守值。
ANIMA_BASE_V1_CONFIG = dict(
    max_img_h=256,         # latent 最大 H(够 2048 输出 / VAE 8x);buffer 大小,运行时取 [:h]
    max_img_w=256,
    max_frames=2,          # 单帧图像 T=1;留 2 余量
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_enable_fps_modulation=False,  # 图像 model 无 fps
    extra_per_block_abs_pos_emb=False,
)


def load_anima_dit_from_single_file(
    path: str | Path,
    *,
    device: str = "cpu",
    dtype: Any = None,
    config_overrides: Optional[dict] = None,
    strict: bool = False,
) -> "Anima":  # noqa: F821 — 字符串前向引用
    """从 anima 单文件 safetensors 加载完整 Anima nn.Module。

    Args:
        path: anima-base-v1.0.safetensors 等单文件路径
        device / dtype: 落到的 device + 计算 dtype(bf16 默认)
        config_overrides: 覆盖 ANIMA_BASE_V1_CONFIG 字段(比如 anima-preview 不同 num_blocks)
        strict: True → load_state_dict(strict=True) 严格匹配(可能挂),False → 允许 missing/unexpected

    Returns:
        Anima 实例,已 `.to(device, dtype)`,已 `.eval()`,可直接 forward。
    """
    import torch  # noqa: PLC0415
    from accelerate import init_empty_weights  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    from .anima import Anima  # noqa: PLC0415

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"anima weights not found: {p}")

    config = dict(ANIMA_BASE_V1_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    # 1) load state_dict + strip 'net.' prefix(anima 真权重 685/685 keys 全带 'net.')。
    raw_sd = load_file(str(p), device="cpu")
    sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k.startswith("net."):
            sd[k[4:]] = v
        else:
            sd[k] = v  # 兜底:未知前缀也试着保留(strict=False 会 ignore)

    # 2) init_empty_weights → Anima(meta tensors,不占 RAM)
    with init_empty_weights():
        model = Anima(**config)

    # 3) load state_dict(assign=True,直接 assign 而非 copy_ —— meta tensor 路径必须)
    missing, unexpected = model.load_state_dict(sd, strict=strict, assign=True)
    if missing or unexpected:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).info(
            "Anima load: missing=%d unexpected=%d (first missing: %s)",
            len(missing), len(unexpected),
            missing[:3] if missing else "(none)",
        )

    # 4) materialize meta params(像 PR-anima-1 build_bridged_text_encoder)
    for name, prm in list(model.named_parameters()):
        if not prm.is_meta:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        target_dtype = dtype if dtype is not None else torch.bfloat16
        setattr(parent, attr, torch.nn.Parameter(
            torch.zeros(prm.shape, dtype=target_dtype), requires_grad=False,
        ))

    # 5) 落 device + dtype
    target_dtype = dtype if dtype is not None else torch.bfloat16
    model = model.to(device, dtype=target_dtype)
    model.eval()
    return model
