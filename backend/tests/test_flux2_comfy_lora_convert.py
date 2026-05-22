"""Flux2 ComfyUI/BFL LoRA 格式预转换 —— 绕 diffusers Flux2LoraLoaderMixin 的
is_kohya 误判(diffusion_model. + lora_down/up 被路由到 lora_unet_ 转换器→零匹配)。

真实样例:用户的 klein_9B_Turbo_r128.safetensors(242 键 diffusion_model.*)经
_convert_non_diffusers_flux2_lora_to_diffusers 转成 306 个 transformer.* diffusers 键。
"""
from __future__ import annotations

import pytest
import torch

from src.services.inference.image_diffusers import _maybe_convert_comfy_flux2_lora

# 实际转换会 import diffusers.loaders.lora_conversion_utils(需真 torch);conftest
# 把 torch 换成 MagicMock(dev_env_gotchas #4)→ 转换用例在 pytest 下 skip,真转换
# 由 standalone smoke(test_flux2_comfy_lora_convert 真文件 242→306 + smoke_granular SMOKE_LORA)验。
# passthrough 用例不 import diffusers(命中前缀/格式检查即返回 None),CI 照跑。
_REAL_TORCH = type(torch).__module__ != "unittest.mock"


def _comfy_flux2_lora() -> dict:
    """最小 ComfyUI/BFL Flux2 LoRA(转换器全消费:img_in 简单映射 + double_blocks
    img_attn.qkv 触发 qkv 三拆)。"""
    r, dim = 8, 64
    return {
        "diffusion_model.img_in.lora_down.weight": torch.zeros(r, dim),
        "diffusion_model.img_in.lora_up.weight": torch.zeros(dim, r),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(r, dim),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": torch.zeros(3 * dim, r),
    }


@pytest.mark.skipif(not _REAL_TORCH, reason="转换 import diffusers 需真 torch(conftest mock 了)")
def test_detects_and_converts_comfy_flux2_lora():
    conv = _maybe_convert_comfy_flux2_lora(_comfy_flux2_lora())
    assert conv is not None
    assert all(k.startswith("transformer.") for k in conv), conv.keys()
    assert any(k.endswith("lora_A.weight") for k in conv)
    assert any(k.endswith("lora_B.weight") for k in conv)
    # qkv 三拆:img_attn.qkv → to_q/to_k/to_v
    assert "transformer.transformer_blocks.0.attn.to_q.lora_B.weight" in conv
    assert "transformer.transformer_blocks.0.attn.to_k.lora_B.weight" in conv
    assert "transformer.transformer_blocks.0.attn.to_v.lora_B.weight" in conv
    # img_in → x_embedder
    assert "transformer.x_embedder.lora_A.weight" in conv


def test_passthrough_already_diffusers_format():
    # 已是 diffusers transformer.* 格式 → 不转(None,走原 load_lora_weights)
    assert _maybe_convert_comfy_flux2_lora({"transformer.x_embedder.lora_A.weight": torch.zeros(2, 2)}) is None


def test_passthrough_kohya_lora_unet_format():
    # 真 Kohya(lora_unet_ 前缀,无 diffusion_model.)→ 不转(交给 diffusers 原生 kohya 路径)
    assert _maybe_convert_comfy_flux2_lora(
        {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(2, 2)}) is None


def test_passthrough_when_no_lora_down_up():
    # diffusion_model. 前缀但无 lora_down/up(非 LoRA / 已转)→ 不转
    assert _maybe_convert_comfy_flux2_lora({"diffusion_model.img_in.weight": torch.zeros(2, 2)}) is None
