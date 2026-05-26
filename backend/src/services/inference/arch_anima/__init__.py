"""nous arch_anima — Cosmos Predict2 + Anima DiT 移植(spec 2026-05-26-anima-port-design)。

PR-anima-1:Cosmos Predict2 building blocks(MiniTrainDIT + 子组件 + 3D RoPE)。
后续 PR-anima-2~7 接 LLMAdapter / text encoder / pipeline 等。
"""
from __future__ import annotations

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

__all__ = [
    "Attention",
    "Block",
    "FinalLayer",
    "GPT2FeedForward",
    "LearnablePosEmbAxis",
    "MiniTrainDIT",
    "PatchEmbed",
    "TimestepEmbedding",
    "Timesteps",
    "VideoPositionEmb",
    "VideoRopePosition3DEmb",
    "apply_rotary_pos_emb",
    "normalize",
]
