"""nous arch_anima — Cosmos Predict2 + Anima DiT 移植(spec 2026-05-26-anima-port-design)。

PR-anima-1:Cosmos Predict2 building blocks(MiniTrainDIT + 子组件 + 3D RoPE)。✅ #161
PR-anima-2:Anima 主类(继承 MiniTrainDIT + LLMAdapter + 1D RoPE TransformerBlock)。✅ #162
PR-anima-3:AnimaTextEncoder(qwen3-0.6b base 单文件 + 可选 t5xxl tokenizer)。✅ 本 PR
后续 PR-anima-4~7 接 pipeline / arch 注册 / 真模型 e2e 等。
"""
from __future__ import annotations

from .anima import Anima, LLMAdapter, RotaryEmbedding
from .position_embedding import (
    LearnablePosEmbAxis,
    VideoPositionEmb,
    VideoRopePosition3DEmb,
    normalize,
)
from .predict2 import (
    Attention,
    Block,
    FinalLayer,
    GPT2FeedForward,
    MiniTrainDIT,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
    apply_rotary_pos_emb,
)
from .text_encoder import AnimaTextEncoder

__all__ = [
    "Anima",
    "AnimaTextEncoder",
    "Attention",
    "Block",
    "FinalLayer",
    "GPT2FeedForward",
    "LLMAdapter",
    "LearnablePosEmbAxis",
    "MiniTrainDIT",
    "PatchEmbed",
    "RotaryEmbedding",
    "TimestepEmbedding",
    "Timesteps",
    "VideoPositionEmb",
    "VideoRopePosition3DEmb",
    "apply_rotary_pos_emb",
    "normalize",
]
