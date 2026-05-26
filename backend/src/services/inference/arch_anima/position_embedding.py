"""Cosmos Predict2 video positional embeddings — ported from ComfyUI.

源:`comfy/ldm/cosmos/position_embedding.py` (NVIDIA Cosmos)
许可:Apache-2.0(原 NVIDIA + ComfyUI)。

Anima 用 `VideoRopePosition3DEmb`(3D RoPE,跨 T/H/W 三轴)和 `LearnablePosEmbAxis`
(per-axis learnable embedding,interpolation="crop")作为 patch token 的位置信号。

Port 改动(vs ComfyUI 原版):
- 无 comfy.* 依赖;纯 torch + einops。
- 保留接口签名(B_T_H_W_C, fps, device, dtype),后续 MiniTrainDIT.forward 调用兼容。
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from einops import rearrange, repeat
from torch import nn


def normalize(x: torch.Tensor, dim: Optional[List[int]] = None, eps: float = 0) -> torch.Tensor:
    """Average-RMS normalization(NVIDIA Cosmos 定义,不同于 L2 normalize)。

    norm 是 sqrt(numel/x.numel) 调整后的 L2 范数 + eps。dim=None 时除 batch 外全部。
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class VideoPositionEmb(nn.Module):
    """Base — 子类实现 `generate_embeddings`。forward 拿 input shape 转发。"""

    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        b_t_h_w_c = x_B_T_H_W_C.shape
        return self.generate_embeddings(b_t_h_w_c, fps=fps, device=device, dtype=dtype)

    def generate_embeddings(
        self,
        b_t_h_w_c: torch.Size,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    """3D RoPE:dim 按 h/w/t 三轴切分,各轴独立频率 + NTK extrapolation factor。

    head_dim 必须能被 6 整除(dim_h = dim_w = dim // 6 * 2;dim_t = 余下)。
    Anima 实测 head_dim=64 → dim_h=dim_w=20, dim_t=24(20+20+24=64)。
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device: Optional[torch.device] = None,
        **kwargs: object,
    ) -> None:
        del kwargs  # 接收 dtype 等无关 kwargs,前向不用
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        # 持久 buffer 但不存 state_dict(persistent=False)—— 是从 device/dim 派生的。
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: dim_h // 2].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: dim_t // 2].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embeddings(
        self,
        b_t_h_w_c: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        del dtype  # 返回 float32(RoPE 频率不掉精度);上游 cast 到 model dtype
        h_ntk = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk
        w_theta = 10000.0 * w_ntk
        t_theta = 10000.0 * t_ntk

        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        _b, t, h, w, _c = b_t_h_w_c
        seq = torch.arange(max(h, w, t), dtype=torch.float32, device=device)
        uniform_fps = (fps is None) or isinstance(fps, (int, float)) or (fps.min() == fps.max())
        assert uniform_fps or _b == 1 or t == 1, (
            "non-uniform fps requires batch=1 (video) or T=1 (image)"
        )
        half_emb_h = torch.outer(seq[:h].to(device=device), h_freqs)
        half_emb_w = torch.outer(seq[:w].to(device=device), w_freqs)
        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:t].to(device=device), t_freqs)
        else:
            half_emb_t = torch.outer(seq[:t].to(device=device) / fps * self.base_fps, t_freqs)

        def _stack_rot(emb: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [torch.cos(emb), -torch.sin(emb), torch.sin(emb), torch.cos(emb)], dim=-1,
            )

        half_emb_h = _stack_rot(half_emb_h)
        half_emb_w = _stack_rot(half_emb_w)
        half_emb_t = _stack_rot(half_emb_t)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=h, w=w),
                repeat(half_emb_h, "h d x -> t h w d x", t=t, w=w),
                repeat(half_emb_w, "w d x -> t h w d x", t=t, h=h),
            ],
            dim=-2,
        )
        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class LearnablePosEmbAxis(VideoPositionEmb):
    """Per-axis learnable pos emb(H/W/T 三轴各自 nn.Parameter)+ crop-only interpolation。

    Anima 用作 extra_per_block_abs_pos_emb(每个 Block 前叠到 hidden states)。
    interpolation="crop":只支持 ≤ len_h/w/t 的 H/W/T(超出报错);extrapolation 留 future。
    """

    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        super().__init__()
        self.interpolation = interpolation
        assert interpolation == "crop", f"unsupported interpolation {interpolation!r}; only 'crop' implemented"

        self.pos_emb_h = nn.Parameter(torch.empty(len_h, model_channels, device=device, dtype=dtype))
        self.pos_emb_w = nn.Parameter(torch.empty(len_w, model_channels, device=device, dtype=dtype))
        self.pos_emb_t = nn.Parameter(torch.empty(len_t, model_channels, device=device, dtype=dtype))

    def generate_embeddings(
        self,
        b_t_h_w_c: torch.Size,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        del fps  # 不参与 — extra_pos_emb 跟 fps 无关
        b, t, h, w, _c = b_t_h_w_c
        emb_h = self.pos_emb_h[:h].to(device=device, dtype=dtype)
        emb_w = self.pos_emb_w[:w].to(device=device, dtype=dtype)
        emb_t = self.pos_emb_t[:t].to(device=device, dtype=dtype)
        emb = (
            repeat(emb_t, "t d -> b t h w d", b=b, h=h, w=w)
            + repeat(emb_h, "h d -> b t h w d", b=b, t=t, w=w)
            + repeat(emb_w, "w d -> b t h w d", b=b, t=t, h=h)
        )
        assert list(emb.shape)[:4] == [b, t, h, w], f"bad shape: {list(emb.shape)[:4]} != {[b, t, h, w]}"
        return normalize(emb, dim=[-1], eps=1e-6)
