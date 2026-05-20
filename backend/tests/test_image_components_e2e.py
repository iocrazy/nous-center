"""PR-4 §9 集成(a): loader→image_generate 描述符流转 + 组件级 L1 缓存命中。"""
from __future__ import annotations

import asyncio
import threading

import pytest

import src.services.nodes.image_components  # noqa: F401 — register loader nodes
from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager
from src.services.nodes.registry import get_node_class


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


class _Collect(PipeChannel):
    def __init__(self):
        self.sent = []

    async def send_message(self, m):
        self.sent.append(m)


async def _build_descriptors():
    """跑 inline loader 节点链产生三描述符(同 WorkflowExecutor 会做的)。"""
    unet = (await get_node_class("image_unet_load")().invoke(
        {"file": "/m/u.safe", "device": "cuda:1", "adapter_arch": "flux2"}, {}))["unet"]
    unet = (await get_node_class("image_lora_apply")().invoke(
        {"lora_file": "style", "lora_path": "/m/loras/style.safe", "strength": 0.8}, {"unet": unet}))["unet"]
    clip = (await get_node_class("image_clip_load")().invoke(
        {"file": "/m/c.safe", "device": "cuda:0", "clip_arch": "flux2"}, {}))["clip"]
    vae = (await get_node_class("image_vae_load")().invoke({"file": "/m/v.safe", "device": "cuda:2"}, {}))["vae"]
    return unet, clip, vae


async def _run_once(state, ch, inputs, task_id):
    node = P.RunNode(task_id=task_id, node_id="g", node_type="image", model_key=None, inputs=inputs)
    state.cancel_flags[task_id] = threading.Event()
    state.run_queue.put_nowait(node)
    task = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.15)
    state.shutdown.set()
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_e2e_descriptor_flow_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))

    mm = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())

    module_loads, assemble = [], []
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: (module_loads.append((spec.kind, tuple((lo.name, lo.strength) for lo in spec.loras))),
                                      {"module": object(), "tokenizer": None})[1])

    class _FakeAdapter:
        async def infer(self, req, **kw):
            from src.services.inference.base import InferenceResult, UsageMeter
            return InferenceResult(media_type="image/png", data=b"\x89PNG",
                                   metadata={"width": req.width, "height": req.height, "seed": req.seed},
                                   usage=UsageMeter(image_count=1, latency_ms=1))

    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: (assemble.append(pc), _FakeAdapter())[1]))

    unet, clip, vae = await _build_descriptors()

    def base_inputs(seed):
        return {"unet": unet, "clip": clip, "vae": vae, "prompt": "a cat",
                "seed": seed, "width": 256, "height": 256, "steps": 4}

    # 第一次:装配一次,三模块各 load 一次
    state = _RunnerState("r", "image", [0, 1, 2], mm)
    ch = _Collect()
    await _run_once(state, ch, base_inputs(42), 1)
    assert [m for m in ch.sent if isinstance(m, P.NodeResult)][-1].status == "completed"
    assert len(assemble) == 1
    assert len(module_loads) == 3

    # 第二次 同描述符 同 seed:combo 缓存命中,不再装配、不再 load
    state2 = _RunnerState("r", "image", [0, 1, 2], mm)
    ch2 = _Collect()
    await _run_once(state2, ch2, base_inputs(42), 2)
    assert len(assemble) == 1
    assert len(module_loads) == 3

    # 第三次 改 LoRA strength:combo 变 → 重装配;但 base 模块(去 LoRA)全命中,不再 load
    unet_b = {**unet, "loras": [{"name": "style", "path": "/m/loras/style.safe", "strength": 0.4}]}
    state3 = _RunnerState("r", "image", [0, 1, 2], mm)
    ch3 = _Collect()
    await _run_once(state3, ch3, {**base_inputs(42), "unet": unet_b}, 3)
    assert len(assemble) == 2          # 新 LoRA 组合 → 新 adapter
    assert len(module_loads) == 3      # 没有任何额外模块 load(clip/vae/unet base 全复用)
