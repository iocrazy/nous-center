"""PR-1 收敛 — Encode Prompt / KSampler 产嵌套描述符(inline, 无 GPU)。

收敛后(spec 2026-05-21 rev 2):Encode/KSampler 不再在主进程 encode/sample,
只累积计划描述符;真正的 encode→denoise→decode 在 image runner 的
ImageSampler.sample() 一把跑完(末端 flux2_vae_decode dispatch 触发)。
flux2_vae_decode 不再是 inline 执行器(走 dispatch),其行为由
test_runner_build_request_granular.py 覆盖。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PKG_DIR = Path(__file__).parents[1] / "nodes" / "flux2-components"


def _load_mod():
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    spec = importlib.util.spec_from_file_location(
        "flux2_components_executor_sampling_test", PKG_DIR / "executor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CLIP = {"_type": "flux2_clip", "type": "flux2",
         "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}]}
_MODEL = {"_type": "flux2_model",
          "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "cuda:1", "dtype": "fp8_e4m3", "adapter_arch": "flux2"},
          "loras": []}


@pytest.mark.asyncio
async def test_encode_prompt_emits_descriptor_no_tensor():
    mod = _load_mod()
    out = await mod.exec_encode_prompt(
        {"text": "a cat", "negative_prompt": ""}, {"clip": _CLIP, "text": "a cat"})
    assert out["conditioning"] == {
        "_type": "flux2_conditioning", "clip": _CLIP, "text": "a cat", "negative": ""}
    assert "prompt_embeds" not in out["conditioning"]  # 不在主进程 encode


@pytest.mark.asyncio
async def test_encode_prompt_rejects_non_clip():
    mod = _load_mod()
    with pytest.raises(RuntimeError, match="CLIP"):
        await mod.exec_encode_prompt({"text": "x"}, {"clip": {"_type": "flux2_model"}})


@pytest.mark.asyncio
async def test_ksampler_emits_descriptor_no_tensor():
    mod = _load_mod()
    out = await mod.exec_ksampler(
        {"width": 768, "height": 768, "steps": 20, "cfg_scale": 4.5, "seed": 42},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "a cat", "negative": ""}})
    lat = out["latent"]
    assert lat["_type"] == "flux2_latent"
    assert lat["model"] is _MODEL
    assert lat["conditioning"]["_type"] == "flux2_conditioning"
    assert (lat["width"], lat["height"], lat["steps"], lat["cfg_scale"], lat["seed"]) == (768, 768, 20, 4.5, 42)
    assert "tensor" not in lat


@pytest.mark.asyncio
async def test_ksampler_seed_blank_is_none():
    mod = _load_mod()
    out = await mod.exec_ksampler(
        {"width": 512, "height": 512, "steps": 1, "cfg_scale": 1.0, "seed": ""},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""}})
    assert out["latent"]["seed"] is None


@pytest.mark.asyncio
async def test_ksampler_rejects_non_model():
    mod = _load_mod()
    with pytest.raises(RuntimeError, match="MODEL"):
        await mod.exec_ksampler({}, {"conditioning": {"_type": "flux2_conditioning"}})


@pytest.mark.asyncio
async def test_ksampler_rejects_non_conditioning():
    mod = _load_mod()
    with pytest.raises(RuntimeError, match="CONDITIONING"):
        await mod.exec_ksampler({}, {"model": _MODEL})


def test_yaml_declares_eight_total_nodes():
    import yaml
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    assert set(cfg["nodes"]) == {
        "flux2_load_checkpoint", "flux2_load_diffusion_model",
        "flux2_load_clip", "flux2_load_vae", "flux2_load_lora",
        "flux2_encode_prompt", "flux2_ksampler", "flux2_vae_decode",
    }


def test_vae_decode_not_in_inline_executors():
    """flux2_vae_decode 收敛后走 dispatch,不在 inline EXECUTORS。"""
    mod = _load_mod()
    assert "flux2_vae_decode" not in mod.EXECUTORS
    assert {"flux2_encode_prompt", "flux2_ksampler"} <= set(mod.EXECUTORS)
