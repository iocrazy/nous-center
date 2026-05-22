"""PR-1 T2:图像引擎选择器 + HF-layout repo 推导(纯逻辑,CI 可跑无 GPU)。

集成路由(get_or_load_image_adapter → ModularImageBackend)由真模型 smoke 验
(plan Task 4/5);此处只测两个纯 helper。
"""
from __future__ import annotations

import pytest

from src.services.inference.component_spec import ComponentSpec
from src.services.model_manager import _modular_repo_from_components, _select_image_engine


def test_engine_default_is_legacy(monkeypatch):
    monkeypatch.delenv("NOUS_IMAGE_ENGINE", raising=False)
    assert _select_image_engine() == "legacy"


def test_engine_modular_from_env(monkeypatch):
    monkeypatch.setenv("NOUS_IMAGE_ENGINE", "Modular")  # 大小写/空格容错
    assert _select_image_engine() == "modular"
    monkeypatch.setenv("NOUS_IMAGE_ENGINE", "  legacy ")
    assert _select_image_engine() == "legacy"


def test_repo_derives_from_hf_layout(tmp_path):
    # HF-layout: <repo>/model_index.json + <repo>/transformer/<weights>
    repo = tmp_path / "Flux2-klein-9B"
    (repo / "transformer").mkdir(parents=True)
    (repo / "model_index.json").write_text("{}")
    unet_file = repo / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    unet_file.write_text("x")
    resolved = {"unet": ComponentSpec(kind="unet", file=str(unet_file), device="cuda:1", dtype="bfloat16")}
    assert _modular_repo_from_components(resolved) == str(repo)


def test_repo_rejects_comfy_single_file(tmp_path):
    # comfy 单文件:diffusion_models/flux/<file>,无 model_index.json → PR-2 桥接
    d = tmp_path / "diffusion_models" / "flux"
    d.mkdir(parents=True)
    f = d / "Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
    f.write_text("x")
    resolved = {"unet": ComponentSpec(kind="unet", file=str(f), device="cuda:1", dtype="bfloat16")}
    with pytest.raises(ValueError, match="PR-2|model_index"):
        _modular_repo_from_components(resolved)
