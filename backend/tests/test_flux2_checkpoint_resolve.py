"""Load Checkpoint → 整模型(diffusers 目录)resolver 产三描述符。

新设计(2026-05-25,对齐 ComfyUI DiffusersLoader):data['file'] = diffusers/<model>/ 目录,
解析 <dir>/{transformer,text_encoder,vae} 各首 .safetensors → MODEL+CLIP+VAE 三件同 device/dtype。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PKG_DIR = Path(__file__).parents[1] / "nodes" / "flux2-components"


def _load_executors():
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    spec = importlib.util.spec_from_file_location(
        "flux2_components_executor_ckpt_test", PKG_DIR / "executor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXECUTORS


def _fake_diffusers_model(tmp_path: Path) -> Path:
    """HF-layout 整模型目录:transformer/ + text_encoder/ + vae/ 各一 .safetensors。"""
    repo = tmp_path / "Flux2-klein-9B"
    for sub, fn in [("transformer", "diffusion_pytorch_model-00001-of-00002.safetensors"),
                    ("text_encoder", "model.safetensors"), ("vae", "diffusion_pytorch_model.safetensors")]:
        (repo / sub).mkdir(parents=True)
        (repo / sub / fn).write_text("x")
    (repo / "model_index.json").write_text("{}")
    return repo


@pytest.mark.asyncio
async def test_checkpoint_resolves_three_descriptors(tmp_path):
    EX = _load_executors()
    repo = _fake_diffusers_model(tmp_path)
    out = await EX["flux2_load_checkpoint"](
        {"file": str(repo), "device": "cuda:0", "weight_dtype": "fp8_e4m3"}, {})
    assert out["model"]["_type"] == "flux2_model"
    assert out["model"]["spec"]["device"] == "cuda:0" and out["model"]["spec"]["dtype"] == "fp8_e4m3"
    assert "transformer/" in out["model"]["spec"]["file"]
    assert out["model"]["loras"] == []
    assert "text_encoder/" in out["clip"]["encoders"][0]["file"]
    assert out["clip"]["encoders"][0]["dtype"] == "fp8_e4m3"
    assert "vae/" in out["vae"]["spec"]["file"] and out["vae"]["spec"]["dtype"] == "fp8_e4m3"


@pytest.mark.asyncio
async def test_checkpoint_defaults(tmp_path):
    EX = _load_executors()
    repo = _fake_diffusers_model(tmp_path)
    out = await EX["flux2_load_checkpoint"]({"file": str(repo)}, {})
    assert out["model"]["spec"]["device"] == "auto"
    assert out["model"]["spec"]["dtype"] == "bfloat16"  # 默认 bf16


@pytest.mark.asyncio
async def test_checkpoint_requires_file():
    EX = _load_executors()
    with pytest.raises(RuntimeError, match="未选整模型|file"):
        await EX["flux2_load_checkpoint"]({}, {})


@pytest.mark.asyncio
async def test_checkpoint_missing_subdir_errors(tmp_path):
    EX = _load_executors()
    repo = tmp_path / "broken"
    (repo / "transformer").mkdir(parents=True)
    (repo / "transformer" / "x.safetensors").write_text("x")  # 缺 text_encoder/vae
    with pytest.raises(RuntimeError, match="text_encoder|缺"):
        await EX["flux2_load_checkpoint"]({"file": str(repo)}, {})
