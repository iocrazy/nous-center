"""HF-layout repo 推导(纯逻辑,CI 可跑无 GPU)。

PR-4 收官:legacy 引擎删除,`_select_image_engine` 选择器随之去掉(image 恒 modular)。
集成路由由真模型 smoke 验。
"""
from __future__ import annotations

import pytest

from src.services.inference.component_spec import ComponentSpec
from src.services.model_manager import _modular_repo_from_components


def test_repo_derives_from_hf_layout(tmp_path):
    # HF-layout: <repo>/model_index.json + <repo>/transformer/<weights>
    repo = tmp_path / "Flux2-klein-9B"
    (repo / "transformer").mkdir(parents=True)
    (repo / "model_index.json").write_text("{}")
    unet_file = repo / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    unet_file.write_text("x")
    resolved = {"diffusion_models": ComponentSpec(kind="diffusion_models", file=str(unet_file), device="cuda:1", dtype="bfloat16")}
    assert _modular_repo_from_components(resolved) == str(repo)


def test_repo_derives_from_clip_when_unet_is_comfy_single_file(tmp_path):
    # PR-2:unet = comfy 量化单文件(无 repo),clip/vae 指向 HF repo → 从 clip 推 repo
    repo = tmp_path / "Flux2-klein-9B"
    (repo / "text_encoder").mkdir(parents=True)
    (repo / "model_index.json").write_text("{}")
    comfy_unet = tmp_path / "diffusion_models" / "flux"
    comfy_unet.mkdir(parents=True)
    unet_f = comfy_unet / "Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
    unet_f.write_text("x")
    clip_f = repo / "text_encoder" / "model.safetensors"
    clip_f.write_text("x")
    resolved = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(unet_f), device="cuda:1", dtype="bfloat16"),
        "clip": ComponentSpec(kind="clip", file=str(clip_f), device="cuda:1", dtype="bfloat16"),
    }
    assert _modular_repo_from_components(resolved) == str(repo)


def test_repo_rejects_comfy_single_file(tmp_path):
    # comfy 单文件:diffusion_models/flux/<file>,无 model_index.json → PR-2 桥接
    d = tmp_path / "diffusion_models" / "flux"
    d.mkdir(parents=True)
    f = d / "Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
    f.write_text("x")
    resolved = {"diffusion_models": ComponentSpec(kind="diffusion_models", file=str(f), device="cuda:1", dtype="bfloat16")}
    with pytest.raises(ValueError, match="PR-2|model_index"):
        _modular_repo_from_components(resolved)
