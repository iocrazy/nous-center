"""Bug 1(节点高亮错位):flux2_vae_decode dispatch 终端按 runner 回传的 stage 把高亮
+ 进度"走链"到对应画布节点(text_encode→Encode Prompt / dit_denoise→KSampler /
vae_decode→VAE Decode),而非一律糊在 VAE Decode 上。"""
from __future__ import annotations

import asyncio

import pytest

from src.runner import protocol as P
from src.services.workflow_executor import WorkflowExecutor


def _chain_workflow():
    return {
        "nodes": [
            {"id": "in_1", "type": "text_input", "data": {"text": "a cat"}},
            {"id": "ld",   "type": "flux2_load_diffusion_model",
             "data": {"file": "/m/u.safe", "device": "cuda:1"}},
            {"id": "clip", "type": "flux2_load_clip", "data": {"file": "/m/c.safe"}},
            {"id": "vae",  "type": "flux2_load_vae",  "data": {"file": "/m/v.safe"}},
            {"id": "enc",  "type": "flux2_encode_prompt", "data": {}},
            {"id": "ksm",  "type": "flux2_ksampler", "data": {"width": 512, "height": 512, "steps": 4, "seed": 1}},
            {"id": "dec",  "type": "flux2_vae_decode", "data": {}},
            {"id": "out",  "type": "image_output", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "in_1", "sourceHandle": "text",  "target": "enc",  "targetHandle": "text"},
            {"id": "e2", "source": "ld",   "sourceHandle": "model", "target": "ksm",  "targetHandle": "model"},
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


def test_compute_image_stage_walk_maps_chain():
    ex = WorkflowExecutor(_chain_workflow())
    sw = ex._compute_image_stage_walk(ex._node_map["dec"])
    assert sw is not None
    assert sw["targets"]["text_encode"] == "enc"
    assert sw["targets"]["dit_denoise"] == "ksm"
    assert sw["targets"]["vae_decode"] == "dec"
    assert sw["initial"] == "ld"  # 加载阶段先点亮 Load Diffusion Model
    # 非 dispatch 节点不参与 stage-walk
    assert ex._compute_image_stage_walk(ex._node_map["enc"]) is None


def test_compute_stage_walk_fallback_to_vae_when_nodes_missing():
    """链上缺 encode/ksampler 时回退到 dispatch 节点本身(不崩,行为同旧版)。"""
    wf = {
        "nodes": [{"id": "dec", "type": "flux2_vae_decode", "data": {}}],
        "edges": [],
    }
    ex = WorkflowExecutor(wf)
    sw = ex._compute_image_stage_walk(ex._node_map["dec"])
    assert sw["targets"]["text_encode"] == "dec"
    assert sw["targets"]["dit_denoise"] == "dec"
    assert sw["initial"] == "dec"


class _StagedClient:
    """run_node 模拟 runner 逐 stage 回传 NodeProgress(text_encode→dit_denoise→vae_decode)。"""
    async def run_node(self, spec, *, on_progress=None, workflow_name=""):
        for stage, step in [("text_encode", 1), ("dit_denoise", 1),
                            ("dit_denoise", 4), ("vae_decode", 1)]:
            if on_progress is not None:
                on_progress(P.NodeProgress(
                    task_id=spec.task_id, node_id=spec.node_id, progress=0.5,
                    stage=stage, step=step, total_steps=4))
        return P.NodeResult(
            task_id=spec.task_id, node_id=spec.node_id, status="completed",
            outputs={"image_url": "/files/x.png", "media_type": "image/png"},
            error=None, duration_ms=1)


@pytest.mark.asyncio
async def test_progress_redirected_to_stage_node():
    """node_progress 的 node_id 按 stage 重定向:denoise→ksampler、encode→encode、
    vae→vae。node_progress 只由 dispatch 发,不被 inline 节点污染,断言干净。"""
    captured: list[dict] = []

    async def on_progress(ev):
        captured.append(ev)

    ex = WorkflowExecutor(_chain_workflow(), on_progress=on_progress,
                          runner_clients={"image": _StagedClient()}, task_id=9)
    await ex.execute()
    await asyncio.sleep(0.05)  # flush _forward_progress create_task'd 的迁移事件

    by_stage = {
        e.get("stage"): e["node_id"]
        for e in captured if e["type"] == "node_progress" and e.get("stage")
    }
    assert by_stage["text_encode"] == "enc"
    assert by_stage["dit_denoise"] == "ksm"   # 关键:denoise 进度落 KSampler,不糊 VAE
    assert by_stage["vae_decode"] == "dec"

    # 走链高亮:dispatch 的 node_start 落加载节点;denoise 阶段点亮 ksampler
    starts = [e["node_id"] for e in captured if e["type"] == "node_start"]
    assert "ld" in starts  # 加载阶段高亮 Load Diffusion Model(而非 VAE)
    assert "ksm" in starts


class _StagedClientWithTimings:
    """同 _StagedClient,但 NodeResult.outputs.meta 带逐组件 stage_latency_ms
    (image_modular 真路径塞的 dit_denoise / vae_decode 各自耗时)。"""
    async def run_node(self, spec, *, on_progress=None, workflow_name=""):
        for stage, step in [("text_encode", 1), ("dit_denoise", 1),
                            ("dit_denoise", 4)]:
            if on_progress is not None:
                on_progress(P.NodeProgress(
                    task_id=spec.task_id, node_id=spec.node_id, progress=0.5,
                    stage=stage, step=step, total_steps=4))
        return P.NodeResult(
            task_id=spec.task_id, node_id=spec.node_id, status="completed",
            outputs={
                "image_url": "/files/x.png", "media_type": "image/png",
                "meta": {"stage_latency_ms": {"dit_denoise": 2800, "vae_decode": 350}},
            },
            error=None, duration_ms=3200)


@pytest.mark.asyncio
async def test_stage_latency_drives_per_node_duration():
    """VAE decode 真计时:KSampler 完成事件 duration_ms = denoise-only(2800),
    VAE Decode(dispatch 终端)完成事件 duration_ms = decode-only(350,非 0)。
    修「KSampler 吃掉 denoise+decode / VAE Decode 恒 0s」。"""
    captured: list[dict] = []

    async def on_progress(ev):
        captured.append(ev)

    ex = WorkflowExecutor(_chain_workflow(), on_progress=on_progress,
                          runner_clients={"image": _StagedClientWithTimings()}, task_id=9)
    await ex.execute()
    await asyncio.sleep(0.05)

    completes = {
        e["node_id"]: e.get("duration_ms")
        for e in captured if e["type"] == "node_complete"
    }
    assert completes.get("ksm") == 2800  # KSampler 显纯 denoise
    assert completes.get("dec") == 350   # VAE Decode 显真 decode 时长(非 0)


@pytest.mark.asyncio
async def test_model_load_stage_broadcast_before_denoise(monkeypatch):
    """Bug 2:dispatch 前广播 stage=model_load 任务进度,任务面板加载阶段显示「加载模型中…」;
    且在第一个 denoise 进度之前。"""
    broadcasts: list[tuple] = []
    from src.api import websocket as ws_mod

    async def _fake_broadcast(task_id, ev):
        broadcasts.append((task_id, ev))

    monkeypatch.setattr(ws_mod.ws_manager, "broadcast_task_progress", _fake_broadcast)

    async def on_progress(_ev):
        pass

    ex = WorkflowExecutor(_chain_workflow(), on_progress=on_progress,
                          runner_clients={"image": _StagedClient()}, task_id=9)
    await ex.execute()
    await asyncio.sleep(0.05)

    stages = [ev.get("stage") for _t, ev in broadcasts]
    assert "model_load" in stages  # 加载阶段有任务级反馈
    ml_idx = next(i for i, (_t, ev) in enumerate(broadcasts) if ev.get("stage") == "model_load")
    dn_idx = next(i for i, (_t, ev) in enumerate(broadcasts) if ev.get("stage") == "dit_denoise")
    assert ml_idx < dn_idx  # model_load 在 denoise 之前
    # 不造假百分比:model_load 不带 step
    ml_ev = broadcasts[ml_idx][1]
    assert "step" not in ml_ev or ml_ev.get("step") is None
