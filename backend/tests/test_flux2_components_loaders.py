"""V1' Lane C / P3.2 — 5 Loader component-node executors.

These tests exercise the executors directly without going through the
WorkflowExecutor; they're cheap (no model load) because Loader nodes only
stash a yaml model_id + bundle kind marker.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


PKG_DIR = Path(__file__).parents[1] / "nodes" / "flux2-components"


def _load_executors():
    """Import the package's executor module via the same mechanism the
    runtime package scanner uses, so we exercise the actual loaded code."""
    import importlib.util
    import sys
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    spec = importlib.util.spec_from_file_location(
        "flux2_components_executor_test", PKG_DIR / "executor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXECUTORS


def test_yaml_declares_loader_nodes():
    """The five loaders this file is about. P3.3 added three more sampling
    nodes (encode_prompt / ksampler / vae_decode) which are covered by
    test_flux2_components_sampling.py."""
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


def test_yaml_node_widgets_use_scanner_driven_model_select():
    """V1' Lane C P4.2: model_key widgets are `model_select, filter: image`
    so the dropdown is populated dynamically from /api/v1/engines?type=image
    by the existing ModelSelectWidget in DeclarativeNode.tsx. Newly-added
    yaml specs (or auto-detected dirs) surface in the dropdown without an
    options-list edit here."""
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    for node_id in ("flux2_load_checkpoint", "flux2_load_diffusion_model",
                    "flux2_load_clip", "flux2_load_vae"):
        widgets = cfg["nodes"][node_id]["widgets"]
        mk = next((w for w in widgets if w["name"] == "model_key"), None)
        assert mk is not None, node_id
        assert mk["widget"] == "model_select", node_id
        assert mk["filter"] == "image", node_id
        # Default fallback to wikeeyang so a freshly-dropped node is usable
        assert mk["default"] == "flux2-klein-9b-true-v2-fp8mixed", node_id


def test_executors_dict_has_five_entries_matching_yaml():
    executors = _load_executors()
    cfg = yaml.safe_load((PKG_DIR / "node.yaml").read_text())
    assert set(executors) == set(cfg["nodes"])


@pytest.mark.asyncio
async def test_load_checkpoint_emits_model_clip_vae_triplet():
    executors = _load_executors()
    out = await executors["flux2_load_checkpoint"]({"model_key": "flux2-klein-9b"}, {})
    assert out["model"] == {"_type": "flux2_model", "model_id": "flux2-klein-9b", "loras": []}
    assert out["clip"] == {"_type": "flux2_clip", "model_id": "flux2-klein-9b"}
    assert out["vae"] == {"_type": "flux2_vae", "model_id": "flux2-klein-9b"}


@pytest.mark.asyncio
async def test_load_checkpoint_defaults_to_wikeeyang_when_model_key_missing():
    """Drag a Loader onto the canvas with no config → still gets the most
    commonly used model rather than crashing."""
    executors = _load_executors()
    out = await executors["flux2_load_checkpoint"]({}, {})
    assert out["model"]["model_id"] == "flux2-klein-9b-true-v2-fp8mixed"


@pytest.mark.asyncio
async def test_load_diffusion_model_only_emits_model():
    executors = _load_executors()
    out = await executors["flux2_load_diffusion_model"]({"model_key": "ernie-image"}, {})
    assert set(out) == {"model"}
    assert out["model"]["_type"] == "flux2_model"


@pytest.mark.asyncio
async def test_load_clip_only_emits_clip():
    executors = _load_executors()
    out = await executors["flux2_load_clip"]({"model_key": "ernie-image"}, {})
    assert set(out) == {"clip"}


@pytest.mark.asyncio
async def test_load_vae_only_emits_vae():
    executors = _load_executors()
    out = await executors["flux2_load_vae"]({"model_key": "ernie-image"}, {})
    assert set(out) == {"vae"}


@pytest.mark.asyncio
async def test_load_lora_appends_to_upstream_model_stack():
    executors = _load_executors()
    upstream_model = {"_type": "flux2_model", "model_id": "flux2-klein-9b", "loras": []}
    out = await executors["flux2_load_lora"](
        {"lora_name": "more_details", "strength": 0.6},
        {"model": upstream_model},
    )
    assert out["model"]["loras"] == [{"name": "more_details", "strength": 0.6}]
    assert out["model"]["model_id"] == "flux2-klein-9b"
    # original upstream must not be mutated
    assert upstream_model["loras"] == []


@pytest.mark.asyncio
async def test_load_lora_chain_accumulates_multiple_loras():
    executors = _load_executors()
    base = {"_type": "flux2_model", "model_id": "flux2-klein-9b", "loras": [
        {"name": "a", "strength": 1.0},
    ]}
    out = await executors["flux2_load_lora"](
        {"lora_name": "b", "strength": 0.5},
        {"model": base},
    )
    assert [lora["name"] for lora in out["model"]["loras"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_load_lora_empty_name_passes_through_unchanged():
    """Blank lora_name = "disabled" loader; matches ComfyUI semantics."""
    executors = _load_executors()
    base = {"_type": "flux2_model", "model_id": "flux2-klein-9b", "loras": []}
    out = await executors["flux2_load_lora"]({"lora_name": "", "strength": 1.0}, {"model": base})
    assert out["model"]["loras"] == []


@pytest.mark.asyncio
async def test_load_lora_rejects_non_model_input():
    """Wired the wrong port into the MODEL slot → fail fast with a clear msg
    instead of producing a malformed bundle the sampler chokes on later."""
    executors = _load_executors()
    with pytest.raises(RuntimeError, match="未连接.*flux2_model"):
        await executors["flux2_load_lora"]({"lora_name": "x"}, {"model": {"_type": "voxcpm2"}})


def test_runtime_package_scanner_picks_up_node_yaml(monkeypatch, tmp_path):
    """Sanity: nodes.scan_packages() honors backend/nodes/flux2-components/."""
    # Use the real package dir layout — scan_packages walks _PACKAGE_DIR
    # which is fixed to backend/nodes/. We just confirm our package shows up.
    from nodes import scan_packages, get_all_definitions, get_all_executors
    scan_packages()
    defs = get_all_definitions()
    execs = get_all_executors()
    for node_type in ("flux2_load_checkpoint", "flux2_load_diffusion_model",
                      "flux2_load_clip", "flux2_load_vae", "flux2_load_lora"):
        assert node_type in defs, f"node.yaml not loaded: {node_type}"
        assert node_type in execs, f"executor not registered: {node_type}"
