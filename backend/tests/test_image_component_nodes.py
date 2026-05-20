"""PR-4: 4 loader 节点输出描述符 dict (inline, 无 GPU)。"""
from __future__ import annotations

import pytest

import src.services.nodes.image_components  # noqa: F401 — 触发 @register 副作用
from src.services.nodes.registry import get_node_class


@pytest.mark.asyncio
async def test_unet_load_emits_descriptor():
    node = get_node_class("image_unet_load")()
    out = await node.invoke(
        {"file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2"},
        {},
    )
    assert out == {"unet": {
        "kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
        "dtype": "bfloat16", "adapter_arch": "flux2", "loras": [],
    }}


@pytest.mark.asyncio
async def test_unet_load_defaults():
    node = get_node_class("image_unet_load")()
    out = await node.invoke({"file": "/m/u.safe"}, {})
    assert out["unet"]["device"] == "auto"
    assert out["unet"]["dtype"] == "bfloat16"
    assert out["unet"]["adapter_arch"] == "flux2"
    assert out["unet"]["loras"] == []


@pytest.mark.asyncio
async def test_clip_and_vae_load():
    clip = await get_node_class("image_clip_load")().invoke(
        {"file": "/m/c.safe", "device": "cuda:0", "clip_arch": "flux2"}, {})
    assert clip == {"clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0",
                             "dtype": "bfloat16", "clip_arch": "flux2"}}
    vae = await get_node_class("image_vae_load")().invoke(
        {"file": "/m/v.safe", "device": "cuda:2"}, {})
    assert vae == {"vae": {"kind": "vae", "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"}}


@pytest.mark.asyncio
async def test_lora_apply_appends():
    upstream = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
                "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "style", "lora_path": "/m/loras/style.safetensors", "strength": 0.8},
        {"unet": upstream},
    )
    assert out["unet"]["loras"] == [{"name": "style", "path": "/m/loras/style.safetensors", "strength": 0.8}]
    assert upstream["loras"] == []


@pytest.mark.asyncio
async def test_lora_apply_chains():
    base = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
            "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    step1 = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "a", "lora_path": "/m/loras/a.safetensors", "strength": 0.8}, {"unet": base})
    step2 = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "b", "lora_path": "/m/loras/b.safetensors", "strength": 0.4}, {"unet": step1["unet"]})
    assert [lo["name"] for lo in step2["unet"]["loras"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_lora_apply_bypass_passthrough():
    upstream = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1",
                "dtype": "bfloat16", "adapter_arch": "flux2", "loras": [{"name": "x", "path": "/p/x", "strength": 1.0}]}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "y", "lora_path": "/p/y", "strength": 0.5, "bypass": True}, {"unet": upstream})
    assert out["unet"] is upstream


@pytest.mark.asyncio
async def test_lora_apply_missing_upstream_raises():
    from src.services.workflow_executor import ExecutionError
    with pytest.raises(ExecutionError, match="unet"):
        await get_node_class("image_lora_apply")().invoke(
            {"lora_file": "x", "lora_path": "/p/x"}, {})


@pytest.mark.asyncio
async def test_lora_apply_accepts_lora_path_only():
    base = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    out = await get_node_class("image_lora_apply")().invoke(
        {"lora_path": "/m/loras/style-xl.safetensors", "strength": 0.7}, {"unet": base})
    lo = out["unet"]["loras"][0]
    assert lo["path"] == "/m/loras/style-xl.safetensors"
    assert lo["name"] == "style-xl.safetensors"
    assert lo["strength"] == 0.7


def test_loader_nodes_are_inline():
    from src.services.node_routing import node_exec_class
    for t in ("image_unet_load", "image_clip_load", "image_vae_load", "image_lora_apply"):
        assert node_exec_class(t) == "inline"
