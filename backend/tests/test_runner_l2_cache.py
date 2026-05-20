"""PR-6: runner L2 cache — deterministic image re-run hits cache (no infer)."""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.runner import protocol as P
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import _RunnerState, _node_executor


class _CountingAdapter:
    def __init__(self): self.calls = 0
    is_loaded = True
    async def infer(self, req, **kw):
        from src.services.inference.base import InferenceResult, UsageMeter
        self.calls += 1
        return InferenceResult(media_type="image/png", data=b"\x89PNG\r\n",
                               metadata={"width": req.width, "height": req.height, "seed": req.seed},
                               usage=UsageMeter(image_count=1, latency_ms=1))


class _MM:
    def __init__(self, adapter): self.adapter = adapter
    async def get_or_load(self, key): return self.adapter
    async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None): return self.adapter


class _Collect(PipeChannel):
    def __init__(self): self.sent = []
    async def send_message(self, m): self.sent.append(m)


def _img_node(task_id, seed, deterministic=True):
    return P.RunNode(task_id=task_id, node_id="g", node_type="image", model_key="m",
                     inputs={"prompt": "a cat", "seed": seed, "width": 64, "height": 64, "steps": 2},
                     is_deterministic=deterministic)


async def _run(state, ch, node):
    state.shutdown.clear()  # 重置，允许同一 state 多次 _run
    state.cancel_flags[node.task_id] = threading.Event()
    state.run_queue.put_nowait(node)
    t = asyncio.create_task(_node_executor(state, ch))
    await asyncio.sleep(0.15)
    state.shutdown.set()
    await asyncio.wait_for(t, timeout=2)


@pytest.mark.asyncio
async def test_deterministic_rerun_hits_l2(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    adapter = _CountingAdapter()
    state = _RunnerState("r", "image", [0], _MM(adapter))

    ch1 = _Collect()
    await _run(state, ch1, _img_node(1, seed=42))
    r1 = [m for m in ch1.sent if isinstance(m, P.NodeResult)][-1]
    assert r1.status == "completed" and adapter.calls == 1
    assert not r1.outputs.get("cached")

    ch2 = _Collect()
    await _run(state, ch2, _img_node(2, seed=42))
    r2 = [m for m in ch2.sent if isinstance(m, P.NodeResult)][-1]
    assert r2.status == "completed" and adapter.calls == 1   # cache hit → no second infer
    assert r2.outputs.get("cached") is True
    assert r2.outputs.get("image_url")


@pytest.mark.asyncio
async def test_random_seed_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_IMAGE_OUTPUTS", str(tmp_path))
    from src.config import get_settings
    monkeypatch.setattr(get_settings(), "ADMIN_SESSION_SECRET", "sek")
    adapter = _CountingAdapter()
    state = _RunnerState("r", "image", [0], _MM(adapter))
    ch1 = _Collect()
    await _run(state, ch1, _img_node(1, seed=7, deterministic=False))
    ch2 = _Collect()
    await _run(state, ch2, _img_node(2, seed=7, deterministic=False))
    assert adapter.calls == 2   # non-deterministic → both infer
