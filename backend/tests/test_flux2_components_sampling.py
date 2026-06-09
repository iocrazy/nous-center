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
async def test_ksampler_no_segment_fields_default(_=None):
    """无分段 widget → 描述符不带分段键(整段采样,零回归)。"""
    mod = _load_mod()
    out = await mod.exec_ksampler(
        {"width": 512, "height": 512, "steps": 12, "cfg_scale": 1.0, "seed": 1},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""}})
    lat = out["latent"]
    for k in ("start_at_step", "end_at_step", "add_noise", "return_with_leftover_noise", "init_latent_ref"):
        assert k not in lat


@pytest.mark.asyncio
async def test_ksampler_segment_widgets_passthrough():
    """PR-B2:base 留噪段 widget(end_at_step + return_with_leftover_noise)透进描述符。
    end_at_step=-1 / "" = 跑到底(不写键);>=0 写键。"""
    mod = _load_mod()
    out = await mod.exec_ksampler(
        {"width": 512, "height": 512, "steps": 12, "cfg_scale": 2.3, "seed": 7,
         "end_at_step": 5, "return_with_leftover_noise": True, "add_noise": True},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""}})
    lat = out["latent"]
    assert lat["end_at_step"] == 5
    assert lat["return_with_leftover_noise"] is True
    assert lat["add_noise"] is True
    assert "start_at_step" not in lat  # 未设 → 不写(默认 0)


@pytest.mark.asyncio
async def test_ksampler_end_at_step_blank_runs_to_end():
    """end_at_step 空 / -1 = 跑到底 → 不写 end_at_step 键(引擎默认 None)。"""
    mod = _load_mod()
    out = await mod.exec_ksampler(
        {"steps": 12, "end_at_step": "", "start_at_step": 5, "add_noise": False},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""}})
    lat = out["latent"]
    assert "end_at_step" not in lat
    assert lat["start_at_step"] == 5
    assert lat["add_noise"] is False


@pytest.mark.asyncio
async def test_ksampler_init_latent_ref_from_port():
    """PR-B2:续采段 —— init_latent 端口接上段 latent_ref(_type=latent_ref)→ 写进描述符。
    非 latent_ref(误连 / 空)= 不写(零回归)。"""
    mod = _load_mod()
    ref = {"_type": "latent_ref", "path": "/nas/latents/x.safetensors", "arch": "z-image", "latent_channels": 16}
    out = await mod.exec_ksampler(
        {"steps": 12, "start_at_step": 5, "add_noise": False},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""},
         "init_latent": ref})
    assert out["latent"]["init_latent_ref"] == ref
    # 误连非 latent_ref → 不写
    out2 = await mod.exec_ksampler(
        {"steps": 12},
        {"model": _MODEL,
         "conditioning": {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""},
         "init_latent": {"_type": "flux2_latent"}})
    assert "init_latent_ref" not in out2["latent"]


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


@pytest.mark.asyncio
async def test_ksampler_rejects_arch_mismatch_anima_dit_flux2_clip():
    """round-2026-06-01:anima DiT + flux2 CLIP 在派发前就抛人话错误,不等 runner
    甩 PyTorch size-mismatch。复现用户 pr3-clip-single 失败(anima 模型接 flux2 CLIP/VAE)。"""
    mod = _load_mod()
    anima_model = {"_type": "flux2_model",
                   "spec": {"kind": "diffusion_models", "file": "/m/anima.safe",
                            "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "anima"},
                   "loras": []}
    flux2_cond = {"_type": "flux2_conditioning", "clip": _CLIP, "text": "x", "negative": ""}
    with pytest.raises(RuntimeError, match="架构不匹配"):
        await mod.exec_ksampler({"steps": 20}, {"model": anima_model, "conditioning": flux2_cond})


@pytest.mark.asyncio
async def test_ksampler_accepts_anima_dit_with_anima_clip():
    """anima DiT + anima/qwen CLIP → 放行。"""
    mod = _load_mod()
    anima_model = {"_type": "flux2_model",
                   "spec": {"kind": "diffusion_models", "file": "/m/anima.safe",
                            "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "anima"},
                   "loras": []}
    anima_clip = {"_type": "flux2_clip", "type": "qwen",
                  "encoders": [{"kind": "clip", "file": "/m/qwen.safe", "dtype": "bfloat16"}]}
    anima_cond = {"_type": "flux2_conditioning", "clip": anima_clip, "text": "x", "negative": ""}
    out = await mod.exec_ksampler({"steps": 20}, {"model": anima_model, "conditioning": anima_cond})
    assert out["latent"]["_type"] == "flux2_latent"


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
