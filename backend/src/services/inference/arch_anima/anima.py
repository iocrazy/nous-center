"""Anima nn.Module — ported from ComfyUI(`comfy/ldm/anima/model.py`)。

接 PR-anima-1 的 `MiniTrainDIT`,补 anima 特有的 LLMAdapter(t5xxl ↔ qwen 文本嵌入桥接)
+ 1D LLM 风 RoPE + Attention/TransformerBlock 内部块。

源:`comfy/ldm/anima/model.py`(CircleStone Labs / Comfy Org 2B DiT)
许可:对应 ComfyUI 项目许可。

## 角色

```
Anima = MiniTrainDIT(Cosmos Predict2 主干)
      + LLMAdapter(把可选 t5xxl 文本嵌入翻译到 qwen3 上下文空间;6 层 transformer)
```

工作流上常见使用方式:`context = qwen3 embeds`(MiniTrainDIT.forward 标准入参)。
当**额外**提供 `t5xxl_ids` kwarg 时:LLMAdapter 把 t5xxl 嵌入投到 qwen 空间,与原 context
拼接 / 替换(见 `preprocess_text_embeds`)。spec 里说「首版可不实现 t5xxl 桥」,但 port
要忠实保留这条路径,留 PR-anima-5 真模型决定要不要走 t5xxl。

## Port 改动(同 PR-anima-1)

- `comfy.operations.Linear/RMSNorm/LayerNorm/Embedding` → `nn.Linear/RMSNorm/LayerNorm/Embedding`。
- `operations.Embedding(out_dtype=...)`(ComfyUI 特有 out_dtype kwarg)→ 在 forward 处用 `.to(dtype)`。
- 删 `transformer_options` 入参。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .predict2 import MiniTrainDIT


# ---------------------- 1D LLM-style RoPE(跟 MiniTrainDIT 的 3D RoPE 是不同算法)----------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_llm_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """LLM-style RoPE 应用(cos/sin 跟 _rotate_half),用于 LLMAdapter 内部 attention。

    跟 `predict2.apply_rotary_pos_emb`(3D RoPE,freqs 复数对)是不同算法 ——
    Anima 同时用两套:外层 MiniTrainDIT 用 3D RoPE,内层 LLMAdapter 用 LLM RoPE。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """LLM 风 1D RoPE 频率生成器(LLaMA / Qwen 等同款)。

    `forward(x, position_ids)` 返回 (cos, sin),用于 _apply_llm_rope。
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos_ids_expanded = position_ids[:, None, :].float()
        # 强制 float32 算 RoPE 频率,避免半精度漂移(LLM RoPE 常规)。
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ pos_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------- Anima 内部 Attention(用于 LLMAdapter,不是 MiniTrainDIT 主干)----------------------


class _AnimaAttention(nn.Module):
    """LLMAdapter 内的 attention 模块,跟 MiniTrainDIT.Attention 区别:
      - 用 1D LLM-style RoPE(query/context 各自 cos/sin 对)
      - 走 attn_mask 路径(target/source mask 4D shape)
      - 输出 proj 叫 o_proj(跟 ComfyUI anima/model.py 对齐;主干那个叫 output_proj)

    Q/K 各自带 RMSNorm,V 不带。
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
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
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_context: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        context_used = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context_used.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context_used).view(kv_shape)).transpose(1, 2)
        v = self.v_proj(context_used).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            q = _apply_llm_rope(q, cos, sin)
            cos, sin = position_embeddings_context
            k = _apply_llm_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn)

    def init_weights(self) -> None:
        """o_proj 零初始化(ComfyUI 训练时用,保证 anima 加载 fresh 时残差路径为 0)。"""
        nn.init.zeros_(self.o_proj.weight)


# ---------------------- Anima TransformerBlock(LLMAdapter 内部 building block)----------------------


class _AnimaTransformerBlock(nn.Module):
    """LLMAdapter 内的 block:self-attn(可选)+ cross-attn + MLP。

    跟 MiniTrainDIT.Block(predict2.Block)区别:
      - 无 AdaLN modulation(纯 norm 风,跟 LLaMA / Qwen 一致)
      - layer_norm=True → LayerNorm,默认 RMSNorm
      - MLP 是 Linear → GELU → Linear(带 bias)
    """

    def __init__(
        self,
        source_dim: int,
        model_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_self_attn: bool = False,
        layer_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.use_self_attn = use_self_attn

        def _norm(dim: int) -> nn.Module:
            return nn.LayerNorm(dim, device=device, dtype=dtype) if layer_norm else nn.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)

        if use_self_attn:
            self.norm_self_attn = _norm(model_dim)
            self.self_attn = _AnimaAttention(
                query_dim=model_dim, context_dim=model_dim,
                n_heads=num_heads, head_dim=model_dim // num_heads,
                device=device, dtype=dtype,
            )

        self.norm_cross_attn = _norm(model_dim)
        self.cross_attn = _AnimaAttention(
            query_dim=model_dim, context_dim=source_dim,
            n_heads=num_heads, head_dim=model_dim // num_heads,
            device=device, dtype=dtype,
        )

        self.norm_mlp = _norm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_context: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(
                normed, mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(
            normed, mask=source_attention_mask, context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self) -> None:
        nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


# ---------------------- LLMAdapter(t5xxl → qwen 桥接)----------------------


class LLMAdapter(nn.Module):
    """6 层 transformer,把 t5xxl 文本 token ids 翻译到 qwen3 嵌入空间。

    Anima 文本编码主路径是 qwen3-0.6b → context 直接喂 MiniTrainDIT。LLMAdapter 是
    **额外**路径:当 anima 工况需要 t5xxl 嵌入(例如 LoRA 训练时)时启用,把 t5xxl token
    ids 经 6 层 cross-attn 投到 qwen 上下文(spec 2026-05-26-anima-port 决策点 3)。

    嵌入 vocab 是 t5xxl 的 32128,经 embed → in_proj(若 model_dim ≠ target_dim) → 6 层
    transformer block(每层带 1D RoPE)→ out_proj → RMSNorm。
    """

    def __init__(
        self,
        source_dim: int = 1024,
        target_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        use_self_attn: bool = True,
        layer_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim, device=device, dtype=dtype)
        self.in_proj = nn.Linear(target_dim, model_dim, device=device, dtype=dtype) if model_dim != target_dim else nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                _AnimaTransformerBlock(
                    source_dim, model_dim,
                    num_heads=num_heads,
                    use_self_attn=use_self_attn, layer_norm=layer_norm,
                    device=device, dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        # ComfyUI 原版用 operations.Embedding(out_dtype=...) — 我们用 nn.Embedding 再 .to(dtype) 替代。
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x, context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


# ---------------------- Anima 主类(继承 MiniTrainDIT)----------------------


class Anima(MiniTrainDIT):
    """Anima 全模型 = MiniTrainDIT + LLMAdapter(可选 t5xxl 桥接)。

    Forward 跟父类 MiniTrainDIT 一致,**额外**支持:
      - `t5xxl_ids`:t5xxl token ids(B, L_t5);提供时走 LLMAdapter 预处理 context。
      - `t5xxl_weights`:可选权重(B, L_t5),逐 token mask LLMAdapter 输出。

    没 t5xxl_ids 时 context 直接喂 MiniTrainDIT(qwen3 标准路径)。
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter(
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: Optional[torch.Tensor],
        t5xxl_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """t5xxl ids + qwen embeds → LLMAdapter 投到 qwen 空间 → 可选权重 mask → pad 到 512。"""
        if text_ids is None:
            return text_embeds
        out = self.llm_adapter(text_embeds, text_ids)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        if out.shape[1] < 512:
            out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        t5xxl_ids: Optional[torch.Tensor] = None,
        t5xxl_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=t5xxl_weights)
        return super().forward(x, timesteps, context, fps=fps, padding_mask=padding_mask)
