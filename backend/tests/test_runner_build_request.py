"""PR-4: _build_request 见 unet/clip/vae 三描述符 → ImageRequest.components。"""
from __future__ import annotations

from src.runner import protocol as P
from src.runner.runner_process import _build_request


def _img_node(inputs):
    return P.RunNode(task_id=1, node_id="g", node_type="image", model_key=None, inputs=inputs)


def test_build_request_components_branch():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "a cat", "steps": 9, "seed": 42, "width": 768, "height": 768,
    })
    req = _build_request(node)
    assert req.components is not None
    assert req.components["unet"].device == "cuda:1"
    assert req.components["clip"].file == "/m/c.safe"
    assert req.prompt == "a cat"
    assert req.steps == 9
    assert req.seed == 42
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_build_request_components_with_loras():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2",
                 "loras": [{"name": "style", "path": "/m/loras/style.safetensors", "strength": 0.8}]},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "x",
    })
    req = _build_request(node)
    assert req.components["unet"].loras[0].name == "style"
    assert req.components["unet"].loras[0].path == "/m/loras/style.safetensors"


def test_build_request_legacy_no_components():
    node = _img_node({"prompt": "x", "steps": 25})
    req = _build_request(node)
    assert req.components is None
    assert req.prompt == "x"


def test_build_request_pipeline_class_override():
    node = _img_node({
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
        "prompt": "x", "pipeline_class": "Flux2Pipeline",
    })
    assert _build_request(node).pipeline_class == "Flux2Pipeline"
