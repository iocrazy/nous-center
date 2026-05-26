"""Latent-to-RGB live preview(ComfyUI 杀手锏的等价实现)。

读 ComfyUI `latent_preview.py` 的 Latent2RGB 路径(默认 method,不依赖任何 TAESD 权重):
对 latent 通道做线性投影(`latent_rgb_factors` 矩阵 + bias)→ 拿到 96px 左右 RGB 缩略图 → JPEG 编码 → base64
data URI。出图过程中节点上叠这个缩略图,实现「看图慢慢长出来」(像 ComfyUI 那种)。

Flux2 特别:128 通道 latent + 2x2 patch unfold → 32 通道 × 2× 空间分辨率,再做 RGB 投影。
factors 从 ComfyUI `comfy/latent_formats.py:Flux2` 直接抄过来(无版权;就是几十个浮点数)。

成本:每步 ~几 ms(投影 + JPEG)+ ~5-10KB / preview 帧。25 步全程 ~50-250KB,WS 带宽 OK。
"""
from __future__ import annotations

import base64
import io
from typing import Any

# Flux2 latent → RGB 投影系数(从 ComfyUI `comfy/latent_formats.py` Flux2 抄过来,2026-05-26):
# - 128 通道 latent → reshape 成 (32, H*2, W*2) → linear(32→3) → RGB。
# 数字就是经验拟合值(几十个 float),不是版权代码。
_FLUX2_RGB_FACTORS = [
    [0.0058, 0.0113, 0.0073], [0.0495, 0.0443, 0.0836], [-0.0099, 0.0096, 0.0644],
    [0.2144, 0.3009, 0.3652], [0.0166, -0.0039, -0.0054], [0.0157, 0.0103, -0.0160],
    [-0.0398, 0.0902, -0.0235], [-0.0052, 0.0095, 0.0109], [-0.3527, -0.2712, -0.1666],
    [-0.0301, -0.0356, -0.0180], [-0.0107, 0.0078, 0.0013], [0.0746, 0.0090, -0.0941],
    [0.0156, 0.0169, 0.0070], [-0.0034, -0.0040, -0.0114], [0.0032, 0.0181, 0.0080],
    [-0.0939, -0.0008, 0.0186], [0.0018, 0.0043, 0.0104], [0.0284, 0.0056, -0.0127],
    [-0.0024, -0.0022, -0.0030], [0.1207, -0.0026, 0.0065], [0.0128, 0.0101, 0.0142],
    [0.0137, -0.0072, -0.0007], [0.0095, 0.0092, -0.0059], [0.0000, -0.0077, -0.0049],
    [-0.0465, -0.0204, -0.0312], [0.0095, 0.0012, -0.0066], [0.0290, -0.0034, 0.0025],
    [0.0220, 0.0169, -0.0048], [-0.0332, -0.0457, -0.0468], [-0.0085, 0.0389, 0.0609],
    [-0.0076, 0.0003, -0.0043], [-0.0111, -0.0460, -0.0614],
]
_FLUX2_RGB_BIAS = [-0.0329, -0.0718, -0.0851]


def _flux2_latent_unpack(latents: Any) -> Any:
    """Flux2 latent 形状归一化:Flux2KleinPipeline 在 denoise loop 里 latents 形状是
    `(B, seq_len, 128)`(packed);要变回 `(B, 32, H*2, W*2)` 才能投 RGB。
    pipeline 用了 2x2 patch unfold,我们这里逆操作。仅用 tensor 方法(.reshape/.permute),无需 import torch。
    """
    if latents.ndim == 3:
        # (B, seq_len=H*W, 128) → 还原空间维。seq_len 假设 H==W(从 cfg infer 不可,这里取 sqrt)。
        b, sl, c = latents.shape
        h = w = int(round(sl**0.5))
        if h * w != sl:
            return None  # 非方形,放弃 preview(回退到无预览)
        latents = latents.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, 128, H, W)
    # latent_rgb_factors_reshape(Flux2 latent_formats.py):
    #   (B, 128, H, W) → (B, 32, 2, 2, H, W).permute(0,1,4,2,5,3) → (B, 32, H*2, W*2)
    b, _c, h, w = latents.shape
    latents = latents.reshape(b, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, 32, h * 2, w * 2)
    return latents


def latent_to_preview_data_uri(latents: Any, *, max_px: int = 96) -> str | None:
    """latents → 96px JPEG data URI。失败返回 None(不阻断推理)。"""
    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        unpacked = _flux2_latent_unpack(latents.detach())
        if unpacked is None:
            return None
        # `linear(x, weight, bias)` 要 weight shape `(out=3, in=32)`;
        # _FLUX2_RGB_FACTORS 原始是 `(32 行 × 3 列)` —— 必须 .T(对齐 ComfyUI Latent2RGB)。
        factors = torch.tensor(_FLUX2_RGB_FACTORS, dtype=unpacked.dtype, device=unpacked.device).T
        bias = torch.tensor(_FLUX2_RGB_BIAS, dtype=unpacked.dtype, device=unpacked.device)
        x = unpacked[0].movedim(0, -1)  # (H, W, 32)
        rgb = torch.nn.functional.linear(x, factors, bias=bias)  # (H, W, 3)
        rgb = ((rgb.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(rgb, mode="RGB")
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        # 任何失败都静默吞 —— preview 是 UX 增强,不能影响推理。
        return None
