"""Cosmos Predict2 MiniTrainDIT — ported from ComfyUI.

源:`comfy/ldm/cosmos/predict2.py` (NVIDIA Cosmos)
许可:Apache-2.0(原 NVIDIA + ComfyUI)。

Anima(CircleStone Labs / Comfy Org 2B DiT)的 base class。完整 7 个组件:
  - `apply_rotary_pos_emb`  3D RoPE 应用
  - `GPT2FeedForward`        MLP
  - `Attention`               self / cross-attn
  - `Timesteps`               sinusoidal time embedding
  - `TimestepEmbedding`       time MLP + AdaLN-LoRA 输出
  - `PatchEmbed`              5D → patch token
  - `FinalLayer`              token → output(逆 patchify 用)
  - `Block`                   self-attn + cross-attn + MLP,带 AdaLN modulation
  - `MiniTrainDIT`            主类(组装 N 个 Block + 3D pos emb)

Port 改动(vs ComfyUI 原版):
- `comfy.operations.Linear/RMSNorm/LayerNorm` → `nn.Linear/RMSNorm/LayerNorm`(torch 2.4+ 原生)。
- `comfy.ldm.modules.attention.optimized_attention` → `F.scaled_dot_product_attention`(torch 2.0+)。
- `comfy.patcher_extension.WrapperExecutor`(forward wrapper)→ 直接 `_forward`(nous 走 diffusers LoRA loader,不需要 ComfyUI patcher)。
- `comfy.ldm.common_dit.pad_to_patch_size` → 内置 `_pad_to_patch_size`(简单 F.pad)。
- 删 `transformer_options` 入参(ComfyUI 特有)。
- 删 `logging.debug` 噪音。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from .position_embedding import LearnablePosEmbAxis, VideoRopePosition3DEmb


# ---------------------- Helpers ----------------------


def _pad_to_patch_size(x: torch.Tensor, patch: Tuple[int, int, int]) -> torch.Tensor:
    """Pad (B, C, T, H, W) → 让 T/H/W 都能被 patch 各维整除(pad 在尾部,常数 0)。

    替代 `comfy.ldm.common_dit.pad_to_patch_size`。patch = (patch_t, patch_h, patch_w)。
    """
    pt, ph, pw = patch
    _b, _c, t, h, w = x.shape
    pad_t = (pt - t % pt) % pt
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    # F.pad 维度顺序倒着来(从最后一维起):W → H → T
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """3D RoPE 应用:t 形状 (..., D) → 视为 (..., D/2, 2) 跟 freqs 复数乘。

    freqs 形状 (L, D/2, 2, 2):由 VideoRopePosition3DEmb.generate_embeddings 提供。
    返回 t.shape,dtype 跟 t 对齐(freqs 是 float32 算更精,casts 出去前 cast 回)。
    """
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def _scaled_dot_product_attention(
    q_BSHD: torch.Tensor, k_BSHD: torch.Tensor, v_BSHD: torch.Tensor,
) -> torch.Tensor:
    """Multi-head attention 替代 ComfyUI `optimized_attention` —— 走 torch 原生 SDPA。

    输入 (B, S, H, D) 三件套;返回 (B, S, H*D)(已 flatten,跟 ComfyUI 行为一致)。
    torch >= 2.0 内部按硬件选最佳(flash-attn / mem-eff / math)。
    """
    # SDPA 要 (B, H, S, D);我们的输入是 (B, S, H, D) — 转置 dim1↔dim2。
    q = q_BSHD.transpose(1, 2)
    k = k_BSHD.transpose(1, 2)
    v = v_BSHD.transpose(1, 2)
    out_BHSD = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, D)
    out_BSHD = out_BHSD.transpose(1, 2).contiguous()  # (B, S, H, D)
    b, s, h, d = out_BSHD.shape
    return out_BSHD.view(b, s, h * d)  # (B, S, H*D)


# ---------------------- Feed-forward ----------------------


class GPT2FeedForward(nn.Module):
    """GELU MLP(无 bias),GPT-2 风格。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


# ---------------------- Attention ----------------------


class Attention(nn.Module):
    """Multi-head attention,自/交叉注意力共用(context_dim=None → self-attn)。

    Q/K 有 RMSNorm(per-head dim);V 走 Identity(原 ComfyUI 设计)。
    self-attn 时 q/k 套 3D RoPE(rope_emb 来自 VideoRopePosition3DEmb)。
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

    def _qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context_used = x if context is None else context
        k = self.k_proj(context_used)
        v = self.v_proj(context_used)
        q, k, v = (
            rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim) for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self._qkv(x, context, rope_emb=rope_emb)
        attn_out = _scaled_dot_product_attention(q, k, v)
        return self.output_dropout(self.output_proj(attn_out))


# ---------------------- Time embedding ----------------------


class Timesteps(nn.Module):
    """Sinusoidal time embedding(扩散模型标配):timesteps (B, T) → (B, T, num_channels)。"""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    """Time embedding → modulation:两层 MLP(可选 AdaLN-LoRA 模式输出 3× hidden)。

    use_adaln_lora=True 时返回 (sample, adaln_lora_B_T_3D);=False 返回 (emb, None)。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adaln_lora: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        out_2 = 3 * out_features if use_adaln_lora else out_features
        self.linear_2 = nn.Linear(out_features, out_2, bias=False, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb  # adaln_lora_B_T_3D = emb;主时间 emb 走 sample 原值
        return emb, None


# ---------------------- Patch embedding ----------------------


class PatchEmbed(nn.Module):
    """5D → patch token:Rearrange (B,C,T,H,W) → (B,T',H',W',C*patch_vol) + Linear。"""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        patch_vol = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size,
            ),
            nn.Linear(patch_vol, out_channels, bias=False, device=device, dtype=dtype),
        )
        self.dim = patch_vol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"PatchEmbed expects 5D (B,C,T,H,W), got {x.dim()}D"
        _b, _c, t, h, w = x.shape
        assert h % self.spatial_patch_size == 0 and w % self.spatial_patch_size == 0, (
            f"H/W ({h},{w}) must be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert t % self.temporal_patch_size == 0, (
            f"T ({t}) must be divisible by temporal_patch_size {self.temporal_patch_size}"
        )
        return self.proj(x)


# ---------------------- Final layer ----------------------


class FinalLayer(nn.Module):
    """DiT 末层:LayerNorm + scale/shift modulation + 投影到 patch_vol × out_channels。"""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        proj_out = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        self.linear = nn.Linear(hidden_size, proj_out, bias=False, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift, scale = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x)


# ---------------------- Transformer Block ----------------------


class Block(nn.Module):
    """DiT block:self-attn + cross-attn + MLP,各带 AdaLN modulation 残差。"""

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        head_dim = x_dim // num_heads

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = Attention(
            x_dim, None, num_heads, head_dim, device=device, dtype=dtype,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, head_dim, device=device, dtype=dtype,
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), device=device, dtype=dtype)

        self.use_adaln_lora = use_adaln_lora

        def _adaln_mod() -> nn.Sequential:
            if use_adaln_lora:
                return nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                    nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
                )
            return nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )

        self.adaln_modulation_self_attn = _adaln_mod()
        self.adaln_modulation_cross_attn = _adaln_mod()
        self.adaln_modulation_mlp = _adaln_mod()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        # AdaLN modulation 3 路:self / cross / mlp,各自 shift/scale/gate(chunk 3)。
        def _mod(modulator: nn.Sequential) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            base = modulator(emb_B_T_D)
            if self.use_adaln_lora and adaln_lora_B_T_3D is not None:
                base = base + adaln_lora_B_T_3D
            return base.chunk(3, dim=-1)

        shift_sa, scale_sa, gate_sa = _mod(self.adaln_modulation_self_attn)
        shift_ca, scale_ca, gate_ca = _mod(self.adaln_modulation_cross_attn)
        shift_mlp, scale_mlp, gate_mlp = _mod(self.adaln_modulation_mlp)

        def _bt11d(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b t d -> b t 1 1 d")

        shift_sa_b, scale_sa_b, gate_sa_b = _bt11d(shift_sa), _bt11d(scale_sa), _bt11d(gate_sa)
        shift_ca_b, scale_ca_b, gate_ca_b = _bt11d(shift_ca), _bt11d(scale_ca), _bt11d(gate_ca)
        shift_mlp_b, scale_mlp_b, gate_mlp_b = _bt11d(shift_mlp), _bt11d(scale_mlp), _bt11d(gate_mlp)

        _b, t, h, w, _d = x_B_T_H_W_D.shape

        def _norm_mod(x: torch.Tensor, norm: nn.Module, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return norm(x) * (1 + scale) + shift

        # 1) self-attn 残差
        normed = _norm_mod(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa_b, shift_sa_b)
        self_attn_out = rearrange(
            self.self_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=t, h=h, w=w,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa_b.to(residual_dtype) * self_attn_out.to(residual_dtype)

        # 2) cross-attn 残差(同样 norm→mod→attn→gate)
        normed = _norm_mod(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca_b, shift_ca_b)
        cross_attn_out = rearrange(
            self.cross_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=t, h=h, w=w,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca_b.to(residual_dtype) * cross_attn_out.to(residual_dtype)

        # 3) MLP 残差
        normed = _norm_mod(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_b, shift_mlp_b)
        mlp_out = self.mlp(normed.to(compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_b.to(residual_dtype) * mlp_out.to(residual_dtype)
        return x_B_T_H_W_D


# ---------------------- Main DiT ----------------------


class MiniTrainDIT(nn.Module):
    """NVIDIA Cosmos 1 base DiT(image/video,3D RoPE)— Anima 继承此类加 LLMAdapter。

    Args(精简版,完整原 ComfyUI 注释见 spec/源):
        max_img_h/w, max_frames:max patch grid 维度
        in_channels / out_channels:输入(latent)/输出 channels
        patch_spatial / patch_temporal:patch 尺寸(spatial 一般 2, temporal 通常 1)
        model_channels / num_blocks / num_heads / mlp_ratio:DiT 主参数
        crossattn_emb_channels:cross-attn context dim(来自 text encoder)
        pos_emb_cls:目前只支 "rope3d"(原 "sincos" 也支但 Anima 不用)
        use_adaln_lora + adaln_lora_dim:AdaLN-LoRA 模式
        rope_*_extrapolation_ratio:RoPE NTK extrapolation
        extra_per_block_abs_pos_emb + extra_*_extrapolation_ratio:LearnablePosEmbAxis
        rope_enable_fps_modulation:fps-aware RoPE
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self._build_pos_embed(device=device, dtype=dtype)
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels, model_channels,
                use_adaln_lora=use_adaln_lora, device=device, dtype=dtype,
            ),
        )

        in_ch_pad = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_ch_pad,
            out_channels=model_channels,
            device=device, dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    device=device, dtype=dtype,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            device=device, dtype=dtype,
        )

        self.t_embedding_norm = nn.RMSNorm(model_channels, eps=1e-6, device=device, dtype=dtype)

    def _build_pos_embed(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ) -> None:
        if self.pos_emb_cls != "rope3d":
            raise ValueError(
                f"pos_emb_cls={self.pos_emb_cls!r} 暂未支持(Anima 用 rope3d);其它请扩 spec。"
            )
        common = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )
        self.pos_embedder = VideoRopePosition3DEmb(
            **common,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **common,
                h_extrapolation_ratio=self.extra_h_extrapolation_ratio,
                w_extrapolation_ratio=self.extra_w_extrapolation_ratio,
                t_extrapolation_ratio=self.extra_t_extrapolation_ratio,
                dtype=dtype,
            )

    def _prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
                )
            else:
                # NEAREST resize 到 (H, W) —— 替代 torchvision.functional.resize 免 torchvision dep。
                # F.interpolate 要 (N, C, H, W);padding_mask 已经是这形状。
                padding_mask = F.interpolate(
                    padding_mask, size=tuple(x_B_C_T_H_W.shape[-2:]), mode="nearest",
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1,
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(
                x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype,
            )

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, None, extra_pos_emb

    def _unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial, p2=self.patch_spatial, t=self.patch_temporal,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: (B, C, T, H, W) latent
            timesteps: (B,) 或 (B, T) 时间步
            context: (B, N, D) cross-attn 文本/conditioning 嵌入
        """
        orig_shape = list(x.shape)
        x = _pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))

        timesteps_B_T = timesteps
        crossattn_emb = context

        x_B_T_H_W_D, rope_emb_L, extra_pos_emb = self._prepare_embedded_sequence(
            x, fps=fps, padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
            self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype),
        )
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape, (
                f"extra_pos_emb shape {extra_pos_emb.shape} != x {x_B_T_H_W_D.shape}"
            )

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L.unsqueeze(1).unsqueeze(0) if rope_emb_L is not None else None,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        # 残差流 fp16 数值不稳;float() 兜底(原 NVIDIA 注释:fp16 clamp 出图有 artifact)。
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)

        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D.to(crossattn_emb.dtype),
            t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )
        out = self._unpatchify(x_B_T_H_W_O)
        return out[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
