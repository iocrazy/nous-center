"""PR-4: _node_executor 走 components 路径 (get_or_load_image_adapter)。"""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor


class _FakeAdapter:
    is_loaded = True

    async def infer(self, req, **kw):
        from src.services.inference.base import InferenceResult, UsageMeter
        return InferenceResult(media_type="image/png", data=b"\x89PNG\r\n",
                               metadata={"width": req.width, "height": req.height, "seed": req.seed},
                               usage=UsageMeter(image_count=1, latency_ms=1))


class _FakeMM:
    def __init__(self):
        self.calls = []

    async def get_or_load_image_adapter(self, components, pipeline_class):
        self.calls.append((tuple(sorted(components)), pipeline_class))
        return _FakeAdapter()

    async def get_or_load(self, key):
        raise AssertionError("legacy get_or_load called on components path")


class _Collect(PipeChannel):
    def __init__(self):
        self.sent = []

    async def send_message(self, m):
        self.sent.append(m)


@pytest.mark.asyncio
async def test_components_path_uses_image_adapter():
    mm = _FakeMM()
    state = _RunnerState("r", "image", [0, 1, 2], mm)
    ch = _Collect()
    node = P.RunNode(
        task_id=5, node_id="g", node_type="image", model_key=None,
        inputs={
            "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
            "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
            "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
            "prompt": "a cat", "seed": 42, "width": 256, "height": 256, "steps": 4,
        })
    state.cancel_flags[5] = threading.Event()
    state.run_queue.put_nowait(node)

    task = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.2)
    state.shutdown.set()
    await asyncio.wait_for(task, timeout=2)

    results = [m for m in ch.sent if isinstance(m, P.NodeResult)]
    assert results and results[-1].status == "completed"
    assert mm.calls == [(("clip", "unet", "vae"), "Flux2KleinPipeline")]


@pytest.mark.asyncio
async def test_components_path_adapter_error_fails_gracefully():
    """A bare exception from get_or_load_image_adapter must become NodeResult
    failed, NOT crash _node_executor (which would hang the workflow)."""
    class _BoomMM:
        async def get_or_load_image_adapter(self, components, pipeline_class):
            raise RuntimeError("scheduler dir not found")
        async def get_or_load(self, key):
            raise AssertionError("legacy path not expected")

    state = _RunnerState("r", "image", [0, 1, 2], _BoomMM())
    ch = _Collect()
    node = P.RunNode(
        task_id=9, node_id="g", node_type="image", model_key=None,
        inputs={
            "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
            "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
            "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
            "prompt": "x", "seed": 1, "width": 64, "height": 64, "steps": 1,
        })
    state.cancel_flags[9] = threading.Event()
    state.run_queue.put_nowait(node)
    task = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.2)
    state.shutdown.set()
    await asyncio.wait_for(task, timeout=2)
    results = [m for m in ch.sent if isinstance(m, P.NodeResult)]
    assert results and results[-1].status == "failed"
    assert "scheduler dir not found" in results[-1].error
