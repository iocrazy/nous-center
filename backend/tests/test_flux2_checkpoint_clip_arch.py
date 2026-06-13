"""Load Checkpoint 的 clip type 跟随整模型 arch(修 #512 后 z-image 整模型被架构校验误杀)。"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_f2_exec", pathlib.Path(__file__).parent.parent / "nodes/flux2-components/executor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_checkpoint_clip_type_follows_arch(tmp_path):
    m = _mod()
    # 造个假整模型目录(三组件各放一个 safetensors)
    for sub in ("transformer", "text_encoder", "vae"):
        d = tmp_path / sub
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x")
    for arch in ("z-image", "flux2", "qwen-edit"):
        out = asyncio.run(m.exec_load_checkpoint(
            {"file": str(tmp_path), "adapter_arch": arch}, {}))
        assert out["clip"]["type"] == arch, f"arch={arch} clip.type 应跟随 arch"
        # 校验链自洽:DiT + 它自己产的 clip 必须过 _check_arch_compat(不被误杀)
        m._check_arch_compat(arch, out["clip"]["type"])
