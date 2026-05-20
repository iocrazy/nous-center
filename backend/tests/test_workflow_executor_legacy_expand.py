"""PR-4 §7.4: 老 image_generate(model_key) dispatch 前 inline 展开成三组件。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.runner import protocol as P
from src.services import workflow_executor as we
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry, ModelSpec
from src.services.model_manager import ModelManager
from src.services.workflow_executor import WorkflowExecutor


class _Capturing:
    def __init__(self):
        self.spec = None

    async def run_node(self, spec, *, workflow_name=""):
        self.spec = spec
        return P.NodeResult(task_id=spec.task_id, node_id=spec.node_id, status="completed",
                            outputs={"image_url": "u"}, error=None, duration_ms=1)


class _Reg(ModelRegistry):
    def __init__(self, spec):
        self._config_path = ""
        self._specs = {spec.id: spec}


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    for sub in ("transformer", "text_encoder", "vae"):
        (root / sub).mkdir(parents=True)
        (root / sub / "model.safetensors").write_bytes(b"x")
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "LOCAL_MODELS_PATH", str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_legacy_image_node_expanded(layout, monkeypatch):
    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000, params={"accepts_lora_archs": ["flux2"]})
    mm = ModelManager(registry=_Reg(spec), allocator=GPUAllocator())
    monkeypatch.setattr(we, "_model_manager", mm)

    wf = {"nodes": [{"id": "g", "type": "image_generate",
                     "data": {"model_key": "flux2-klein-9b", "prompt": "x", "seed": 1,
                              "loras": [{"name": "style", "strength": 0.6}]}}],
          "edges": []}
    client = _Capturing()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=3)
    await ex._dispatch_node(ex._node_map["g"], {})

    inp = client.spec.inputs
    assert {"unet", "clip", "vae"} <= set(inp)
    assert inp["unet"]["loras"][0]["name"] == "style"
    assert inp["unet"]["device"] == "auto"
    assert Path(inp["clip"]["file"]).parent.name == "text_encoder"


@pytest.mark.asyncio
async def test_new_format_not_expanded(layout, monkeypatch):
    mm = ModelManager(registry=_Reg(ModelSpec(id="x", model_type="image", adapter_class="a",
                                              paths={"main": "Flux2-klein-9B"}, vram_mb=1)),
                      allocator=GPUAllocator())
    monkeypatch.setattr(we, "_model_manager", mm)
    wf = {"nodes": [{"id": "g", "type": "image_generate", "data": {"prompt": "x"}}], "edges": []}
    client = _Capturing()
    ex = WorkflowExecutor(wf, runner_clients={"image": client}, task_id=3)
    upstream_unet = {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []}
    await ex._dispatch_node(ex._node_map["g"], {
        "unet": upstream_unet,
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    })
    assert client.spec.inputs["unet"] is upstream_unet
