"""nous arch_anima — Cosmos Predict2 + Anima DiT 移植(spec 2026-05-26-anima-port-design)。

PR-anima-1:Cosmos Predict2 building blocks。✅ #161
PR-anima-2:Anima 主类 + LLMAdapter。✅ #162
PR-anima-3:AnimaTextEncoder wrapper。✅ #163
PR-anima-4:权重加载器(真 2.09B 跑通)。✅ #164
PR-anima-5(p1):Qwen3-0.6B-Base config bundled。✅ #165
PR-anima-5(p2):AnimaPipeline 装配类(本 PR)。
后续 PR-anima-6~7 接 arch 注册 / 真模型 e2e。
"""
from __future__ import annotations

from .anima import Anima, LLMAdapter, RotaryEmbedding
from .load import ANIMA_BASE_V1_CONFIG, load_anima_dit_from_single_file
from .pipeline import AnimaPipeline
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
    "ANIMA_BASE_V1_CONFIG",
    "Anima",
    "AnimaPipeline",
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
    "load_anima_dit_from_single_file",
    "normalize",
]
