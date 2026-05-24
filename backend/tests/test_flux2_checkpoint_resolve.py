"""PR-1 T6: Load Checkpoint → model_key resolver 产三描述符(复用 expand_legacy_image_spec)。

单合并 spec → unet/clip/vae 三文件,三件同 device(便捷单卡入口)。device/weight_dtype
来自节点控件;文件路径由 expand_legacy_image_spec 从 ModelSpec 解析。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def stub_mm(monkeypatch):
    from src.services import workflow_executor as we
    fake_spec = SimpleNamespace(paths={"main": "/tmp/fakemodel"},
                                params={"accepts_lora_archs": ["flux2"]})
    fake_mm = SimpleNamespace(_registry=SimpleNamespace(get=lambda k: fake_spec))
    monkeypatch.setattr(we, "_model_manager", fake_mm)
    return fake_mm


@pytest.mark.asyncio
async def test_checkpoint_resolves_three_descriptors(stub_mm):
    EX = _load_executors()
    out = await EX["flux2_load_checkpoint"](
        {"model_key": "flux2-klein-9b", "device": "cuda:0", "weight_dtype": "fp8_e4m3"}, {})
    assert out["model"]["_type"] == "flux2_model"
    assert out["model"]["spec"]["device"] == "cuda:0"
    assert out["model"]["spec"]["dtype"] == "fp8_e4m3"
    assert "transformer" in out["model"]["spec"]["file"]
    assert out["model"]["loras"] == []
    assert out["clip"]["_type"] == "flux2_clip"
    assert out["clip"]["encoders"][0]["dtype"] == "fp8_e4m3"
    assert "text_encoder" in out["clip"]["encoders"][0]["file"]
    assert out["vae"]["_type"] == "flux2_vae"
    assert out["vae"]["spec"]["dtype"] == "fp8_e4m3"
    assert "vae" in out["vae"]["spec"]["file"]


@pytest.mark.asyncio
async def test_checkpoint_defaults(stub_mm):
    EX = _load_executors()
    out = await EX["flux2_load_checkpoint"]({}, {})
    assert out["model"]["spec"]["device"] == "auto"
    assert out["model"]["spec"]["dtype"] == "bfloat16"  # 默认 bf16(非 fp32-default)


@pytest.mark.asyncio
async def test_checkpoint_requires_model_manager(monkeypatch):
    from src.services import workflow_executor as we
    monkeypatch.setattr(we, "_model_manager", None)
    EX = _load_executors()
    with pytest.raises(RuntimeError, match="ModelManager"):
        await EX["flux2_load_checkpoint"]({"model_key": "x"}, {})
