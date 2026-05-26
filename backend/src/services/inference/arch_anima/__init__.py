"""nous arch_anima — Cosmos Predict2 + Anima DiT 移植(spec 2026-05-26-anima-port-design)。

PR-anima-1:Cosmos Predict2 building blocks(MiniTrainDIT + 子组件 + 3D RoPE)。✅ #161
PR-anima-2:Anima 主类(继承 MiniTrainDIT + LLMAdapter + 1D RoPE)。✅ #162
PR-anima-3:AnimaTextEncoder wrapper(qwen3-0.6b base + 可选 t5xxl tokenizer)。✅ #163
PR-anima-4:跳过(anima-base-v1.0 全 bf16 无量化,标准 nn.Linear 够);
            落地加载器 load_anima_dit_from_single_file(strip 'net.' prefix)。✅ 本 PR
后续 PR-anima-5~7 接 AnimaPipeline 装配 / arch 注册 / 真模型 e2e 等。
"""
from __future__ import annotations

from .anima import Anima, LLMAdapter, RotaryEmbedding
from .load import ANIMA_BASE_V1_CONFIG, load_anima_dit_from_single_file
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
