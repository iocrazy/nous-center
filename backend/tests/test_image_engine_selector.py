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


def test_single_file_flux2_uses_bundled_config(tmp_path, monkeypatch):
    """**PR-B**:flux2 全单文件 → 仓内 bundled config(backend/configs/image_arch/flux2/),
    **不再**依赖 LOCAL_MODELS_PATH/image/diffusers/Flux2-klein-9B 参考整模型(18GB)。"""
    from unittest.mock import MagicMock
    d = tmp_path / "image" / "diffusion_models" / "flux"
    d.mkdir(parents=True)
    f = d / "Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
    f.write_text("x")
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(tmp_path)  # **无任何 diffusers/ 参考** —— 仍能跑
    monkeypatch.setattr("src.config.get_settings", lambda: settings)

    resolved = {"diffusion_models": ComponentSpec(
        kind="diffusion_models", file=str(f), device="cuda:1", dtype="bfloat16", adapter_arch="flux2")}
    result = _modular_repo_from_components(resolved)
    assert result is not None
    assert result.endswith("configs/image_arch/flux2"), f"应返回仓内 bundle,得到 {result!r}"


def test_single_file_unknown_arch_falls_back_to_local_diffusers(tmp_path, monkeypatch):
    """未知架构(无 bundle)→ fallback 扫 LOCAL_MODELS_PATH/image/diffusers/ 找匹配整模型。
    向后兼容老用户。"""
    from unittest.mock import MagicMock
    d = tmp_path / "image" / "diffusion_models"
    d.mkdir(parents=True)
    f = d / "ernie.safetensors"
    f.write_text("x")
    ref = tmp_path / "image" / "diffusers" / "ERNIE-Image"
    ref.mkdir(parents=True)
    (ref / "model_index.json").write_text('{"_class_name": "ErnieImagePipeline"}')
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(tmp_path)
    monkeypatch.setattr("src.config.get_settings", lambda: settings)

    resolved = {"diffusion_models": ComponentSpec(
        kind="diffusion_models", file=str(f), device="cuda:1", dtype="bfloat16", adapter_arch="ernie")}
    assert _modular_repo_from_components(resolved) == str(ref)


def test_single_file_unknown_arch_no_bundle_no_local_raises(tmp_path, monkeypatch):
    """未知架构 + 无 bundle + 无 LOCAL_MODELS_PATH 参考 → 清晰报错。"""
    from unittest.mock import MagicMock
    d = tmp_path / "image" / "diffusion_models"
    d.mkdir(parents=True)
    f = d / "x.safetensors"
    f.write_text("x")
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(tmp_path)
    monkeypatch.setattr("src.config.get_settings", lambda: settings)
    resolved = {"diffusion_models": ComponentSpec(
        kind="diffusion_models", file=str(f), device="cuda:1", dtype="bfloat16", adapter_arch="ernie")}
    with pytest.raises(ValueError, match="参考整模型|model_index"):
        _modular_repo_from_components(resolved)
