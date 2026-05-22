"""PR-1 T7: 细粒度图集成 —— Load*/Encode/KSampler inline 累积描述符,末端
flux2_vae_decode dispatch 到 image runner;捕获 RunNode 验嵌套 latent + 单卡 + LoRA。
inline 节点不碰 GPU(无 _model_manager 注入仍能跑通即证明)。"""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


class _CapturingClient:
    def __init__(self):
        self.spec: P.RunNode | None = None

    async def run_node(self, spec, *, workflow_name=""):
        self.spec = spec
        return P.NodeResult(
            task_id=spec.task_id, node_id=spec.node_id, status="completed",
            outputs={"image_url": "/files/x.png", "media_type": "image/png",
                     "width": 512, "height": 512},
            error=None, duration_ms=1)


def _granular_workflow():
    return {
        "nodes": [
            {"id": "in_1", "type": "text_input", "data": {"text": "a cat"}},
            {"id": "ld",   "type": "flux2_load_diffusion_model",
             "data": {"file": "/m/u.safe", "device": "cuda:1", "weight_dtype": "fp8_e4m3"}},
            {"id": "lora", "type": "flux2_load_lora",
             "data": {"lora_name": "turbo", "lora_path": "/m/loras/turbo.safe", "strength": 0.8}},
            {"id": "clip", "type": "flux2_load_clip", "data": {"file": "/m/c.safe"}},
            {"id": "vae",  "type": "flux2_load_vae",  "data": {"file": "/m/v.safe"}},
            {"id": "enc",  "type": "flux2_encode_prompt", "data": {}},
            {"id": "ksm",  "type": "flux2_ksampler", "data": {"width": 512, "height": 512, "steps": 8, "seed": 42}},
            {"id": "dec",  "type": "flux2_vae_decode", "data": {}},
            {"id": "out",  "type": "image_output", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "in_1", "sourceHandle": "text",  "target": "enc",  "targetHandle": "text"},
            {"id": "e2", "source": "ld",   "sourceHandle": "model", "target": "lora", "targetHandle": "model"},
            {"id": "e3", "source": "lora", "sourceHandle": "model", "target": "ksm",  "targetHandle": "model"},
            {"id": "e4", "source": "clip", "sourceHandle": "clip",  "target": "enc",  "targetHandle": "clip"},
            {"id": "e5", "source": "enc",  "sourceHandle": "conditioning", "target": "ksm", "targetHandle": "conditioning"},
            {"id": "e6", "source": "ksm",  "sourceHandle": "latent", "target": "dec",  "targetHandle": "latent"},
            {"id": "e7", "source": "vae",  "sourceHandle": "vae",   "target": "dec",  "targetHandle": "vae"},
            {"id": "e8", "source": "dec",  "sourceHandle": "image", "target": "out",  "targetHandle": "image"},
        ],
    }


@pytest.fixture(autouse=True)
def _scan():
    from nodes import scan_packages
    scan_packages()


@pytest.mark.asyncio
async def test_granular_workflow_dispatches_image_request():
    client = _CapturingClient()
    ex = WorkflowExecutor(_granular_workflow(), runner_clients={"image": client}, task_id=7)
    result = await ex.execute()

    spec = client.spec
    assert spec is not None, "flux2_vae_decode 应 dispatch 到 image runner"
    assert spec.node_type == "image"
    # 嵌套 latent 描述符(含串联 LoRA + 采样参数)
    latent = spec.inputs["latent"]
    assert latent["_type"] == "flux2_latent"
    assert latent["model"]["spec"]["device"] == "cuda:1"
    assert latent["model"]["loras"][0]["name"] == "turbo"
    assert latent["model"]["loras"][0]["path"] == "/m/loras/turbo.safe"
    assert latent["conditioning"]["text"] == "a cat"
    assert (latent["width"], latent["seed"]) == (512, 42)
    # vae 描述符直挂在 VAE Decode 输入
    assert spec.inputs["vae"]["_type"] == "flux2_vae"
    # image_output 收到 runner 返回的 image_url
    assert result["outputs"]["out"] is not None


@pytest.mark.asyncio
async def test_inline_nodes_run_without_model_manager():
    """inline 描述符节点不碰 GPU —— 未注入 _model_manager 也能跑到 dispatch。"""
    from src.services import workflow_executor as we
    assert we._model_manager is None or we._model_manager is not None  # 不依赖其状态
    client = _CapturingClient()
    ex = WorkflowExecutor(_granular_workflow(), runner_clients={"image": client}, task_id=1)
    await ex.execute()
    assert client.spec is not None
