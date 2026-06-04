"""PR-1 收敛 — flux2 loader 节点产嵌套描述符(inline, 无 GPU)。

收敛后(spec 2026-05-21 rev 2):Load Diffusion/CLIP/VAE 各选 file + weight_dtype
(Load Diffusion 另加 device),产 ComponentSpec 形态的嵌套描述符。Load Checkpoint
暂留 model_key 旧形态(PR-1 Task 6 改成 resolver);Load LoRA 串联并带 path。
执行器不再在主进程碰 GPU。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


PKG_DIR = Path(__file__).parents[1] / "nodes" / "flux2-components"


def _load_executors():
    """Import the package's executor module via the same mechanism the
    runtime package scanner uses, so we exercise the actual loaded code."""
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    spec = importlib.util.spec_from_file_location(
        "flux2_components_executor_test", PKG_DIR / "executor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXECUTORS


def test_yaml_declares_loader_nodes():
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    nodes = cfg["nodes"]
    loader_nodes = {
        "flux2_load_checkpoint",
        "flux2_load_diffusion_model",
        "flux2_load_clip",
        "flux2_load_vae",
        "flux2_load_lora",
    }
    assert loader_nodes <= set(nodes)


def test_load_diffusion_widgets_have_file_dtype_device():
    """收敛后 Load Diffusion Model 有 file(component_select) + weight_dtype + device
    + adapter_arch,对齐 ComfyUI UNETLoaderMultiGPU。"""
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    node = cfg["nodes"]["flux2_load_diffusion_model"]
    assert node.get("componentRole") == "diffusion_models"
    by_name = {w["name"]: w for w in node["widgets"]}
    assert by_name["file"]["widget"] == "component_select"
    assert by_name["file"]["role"] == "diffusion_models"
    assert "weight_dtype" in by_name
    assert "default" in [o if isinstance(o, str) else o for o in by_name["weight_dtype"]["options"]]
    assert by_name["device"]["widget"] == "select"


def test_load_vae_widgets_have_file_dtype():
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    vae = cfg["nodes"]["flux2_load_vae"]
    assert vae.get("componentRole") == "vae"
    vae_w = {w["name"]: w for w in vae["widgets"]}
    assert vae_w["file"]["widget"] == "component_select" and vae_w["file"]["role"] == "vae"
    assert "weight_dtype" in vae_w


def test_load_clip_widgets_clip_stack_and_type():
    """PR-3 动态多 CLIP:Load CLIP 用 clip_stack(增删条目)+ type(架构),
    不再是单 file(节点级 componentRole 移除,改 clip_stack 每行状态点)。"""
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    clip = cfg["nodes"]["flux2_load_clip"]
    assert "componentRole" not in clip
    clip_w = {w["name"]: w for w in clip["widgets"]}
    assert clip_w["clips"]["widget"] == "clip_stack"
    assert clip_w["type"]["widget"] == "select"
    assert "flux2" in clip_w["type"]["options"] and "flux1" in clip_w["type"]["options"]


def test_executors_dict_is_yaml_nodes_minus_dispatch():
    """flux2_vae_decode 收敛后走 dispatch(runner 执行),不在 inline EXECUTORS。"""
    executors = _load_executors()
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    assert set(executors) == set(cfg["nodes"]) - {"flux2_vae_decode"}


# --- Load Diffusion / CLIP / VAE → 嵌套描述符 ---------------------------------


@pytest.mark.asyncio
async def test_load_diffusion_descriptor():
    executors = _load_executors()
    out = await executors["flux2_load_diffusion_model"](
        {"file": "/m/u.safe", "device": "cuda:1", "weight_dtype": "fp8_e4m3", "adapter_arch": "flux2"}, {})
    assert out["model"] == {
        "_type": "flux2_model",
        "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "cuda:1",
                 "dtype": "fp8_e4m3", "adapter_arch": "flux2"},
        "loras": [],
        "offload": "none",
    }


@pytest.mark.asyncio
async def test_load_diffusion_defaults():
    executors = _load_executors()
    out = await executors["flux2_load_diffusion_model"]({"file": "/m/u.safe"}, {})
    s = out["model"]["spec"]
    # 默认 dtype = bfloat16(非 "default"→fp32,4× 慢);device 仍 auto。
    assert s["device"] == "auto" and s["dtype"] == "bfloat16" and s["adapter_arch"] == "flux2"
    assert out["model"]["loras"] == []


@pytest.mark.asyncio
async def test_load_clip_single_via_clips():
    executors = _load_executors()
    out = await executors["flux2_load_clip"]({"clips": [{"file": "/m/c.safe", "weight_dtype": "default"}]}, {})
    assert out["clip"] == {
        "_type": "flux2_clip", "type": "flux2",
        "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}],
        "device": "auto", "offload": "none",
    }


@pytest.mark.asyncio
async def test_load_clip_multi_encoder():
    executors = _load_executors()
    out = await executors["flux2_load_clip"]({"type": "flux1", "clips": [
        {"file": "/m/clipL.safe", "weight_dtype": "bfloat16"},
        {"file": "/m/t5.safe", "weight_dtype": "fp8_e4m3"},
    ]}, {})
    assert out["clip"] == {"_type": "flux2_clip", "type": "flux1", "encoders": [
        {"kind": "clip", "file": "/m/clipL.safe", "dtype": "bfloat16"},
        {"kind": "clip", "file": "/m/t5.safe", "dtype": "fp8_e4m3"},
    ], "device": "auto", "offload": "none"}


@pytest.mark.asyncio
async def test_load_clip_legacy_single_file_fallback():
    """PR-1/PR-2 期存的单 file 格式仍可解析(back-compat)。"""
    executors = _load_executors()
    out = await executors["flux2_load_clip"]({"file": "/m/c.safe", "weight_dtype": "bfloat16"}, {})
    assert out["clip"]["encoders"] == [{"kind": "clip", "file": "/m/c.safe", "dtype": "bfloat16"}]


@pytest.mark.asyncio
async def test_load_vae_descriptor():
    executors = _load_executors()
    out = await executors["flux2_load_vae"]({"file": "/m/v.safe", "weight_dtype": "bfloat16"}, {})
    assert out["vae"] == {"_type": "flux2_vae", "spec": {
        "kind": "vae", "file": "/m/v.safe", "dtype": "bfloat16", "device": "auto", "offload": "none"}}


# --- Load LoRA 串联(带 path)-------------------------------------------------


@pytest.mark.asyncio
async def test_load_lora_appends_with_path():
    executors = _load_executors()
    base = {"_type": "flux2_model",
            "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "cuda:1", "dtype": "fp8_e4m3", "adapter_arch": "flux2"},
            "loras": []}
    out = await executors["flux2_load_lora"](
        {"lora_name": "more_details", "lora_path": "/m/loras/more.safe", "strength": 0.6},
        {"model": base})
    assert out["model"]["loras"] == [{"name": "more_details", "path": "/m/loras/more.safe", "strength": 0.6}]
    assert base["loras"] == []  # 上游不被改


@pytest.mark.asyncio
async def test_load_lora_chain_accumulates():
    executors = _load_executors()
    base = {"_type": "flux2_model", "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "cuda:1",
            "dtype": "fp8_e4m3", "adapter_arch": "flux2"}, "loras": []}
    s1 = await executors["flux2_load_lora"]({"lora_name": "a", "lora_path": "/m/a.safe", "strength": 0.8}, {"model": base})
    s2 = await executors["flux2_load_lora"]({"lora_name": "b", "lora_path": "/m/b.safe", "strength": 0.4}, {"model": s1["model"]})
    assert [lora["name"] for lora in s2["model"]["loras"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_load_lora_empty_name_passes_through():
    executors = _load_executors()
    base = {"_type": "flux2_model", "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "auto",
            "dtype": "default", "adapter_arch": "flux2"}, "loras": []}
    out = await executors["flux2_load_lora"]({"lora_name": "", "strength": 1.0}, {"model": base})
    assert out["model"]["loras"] == []


@pytest.mark.asyncio
async def test_load_lora_rejects_non_model_input():
    executors = _load_executors()
    with pytest.raises(RuntimeError, match="flux2_model"):
        await executors["flux2_load_lora"]({"lora_name": "x"}, {"model": {"_type": "voxcpm2"}})


# Load Checkpoint resolver(model_key→三描述符)由 test_flux2_checkpoint_resolve.py 覆盖。


def test_runtime_package_scanner_picks_up_node_yaml():
    from nodes import scan_packages, get_all_definitions, get_all_executors
    scan_packages()
    defs = get_all_definitions()
    execs = get_all_executors()
    for node_type in ("flux2_load_checkpoint", "flux2_load_diffusion_model",
                      "flux2_load_clip", "flux2_load_vae", "flux2_load_lora"):
        assert node_type in defs, f"node.yaml not loaded: {node_type}"
        assert node_type in execs, f"executor not registered: {node_type}"
    # vae_decode 仍在 definitions(画布有这节点)但不在 inline executors(走 dispatch)
    assert "flux2_vae_decode" in defs
    assert "flux2_vae_decode" not in execs
