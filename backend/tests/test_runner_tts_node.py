"""Lane F: TTS runner 迁入测试 —— FakeTTSAdapter 单测 + runner 子进程 TTS 路径。

零 GPU / 零真模型：FakeTTSAdapter 继承 TTSEngine 但 synthesize 返回固定 wav
bytes，不 import torch。runner 子进程测试起真 multiprocessing.Process。
"""
import asyncio
import multiprocessing as mp
import uuid
from pathlib import Path

import pytest

from src.runner import protocol as P
from src.runner.fake_tts_adapter import FakeTTSAdapter
from src.runner.pipe_channel import PipeChannel
from src.runner.runner_process import runner_main
from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceResult,
    MediaModality,
)
from src.workers.tts_engines.base import TTSEngine

_SPAWN = mp.get_context("spawn")
_FIXTURE = str(Path(__file__).parent / "fixtures" / "runner_tts_models.yaml")


def _audio_req(text: str = "你好世界") -> AudioRequest:
    return AudioRequest(request_id=str(uuid.uuid4()), text=text)


def test_fake_tts_adapter_is_inference_adapter():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    assert isinstance(a, InferenceAdapter)
    assert isinstance(a, TTSEngine)
    assert a.modality is MediaModality.AUDIO


@pytest.mark.asyncio
async def test_fake_tts_adapter_load_and_infer():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    assert a.is_loaded
    result = await a.infer(_audio_req())
    assert isinstance(result, InferenceResult)
    assert result.media_type == "audio/wav"
    assert result.data  # 非空 wav bytes
    assert result.metadata["sample_rate"] == 24000
    assert result.metadata["format"] == "wav"
    assert result.usage.audio_seconds is not None


@pytest.mark.asyncio
async def test_fake_tts_adapter_infer_before_load_raises():
    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    with pytest.raises(RuntimeError):
        await a.infer(_audio_req())


@pytest.mark.asyncio
async def test_fake_tts_adapter_rejects_non_audio_request():
    """TTSEngine.infer 对非 AudioRequest 抛 TypeError —— FakeTTSAdapter 继承此行为。"""
    from src.services.inference.base import ImageRequest

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    with pytest.raises(TypeError):
        await a.infer(ImageRequest(request_id="x", prompt="a cat"))


# ---- runner 子进程 TTS 路径（真 multiprocessing.Process）----


def _spawn_tts_runner(group_id: str = "tts", gpus: tuple[int, ...] = (3,)):
    """起一个 fake TTS runner 子进程 —— yaml fixture 把 adapter_class 写死成
    FakeTTSAdapter，所以 fake_adapter=False 也能跑零 GPU 测试。"""
    parent_conn, child_conn = _SPAWN.Pipe()
    proc = _SPAWN.Process(
        target=runner_main,
        args=(group_id, list(gpus), child_conn),
        kwargs={"models_yaml_path": _FIXTURE, "fake_adapter": False},
        daemon=True,
    )
    proc.start()
    child_conn.close()  # 主进程侧不用 child 端
    return proc, PipeChannel(parent_conn)


async def _recv(ch: PipeChannel, timeout: float = 10.0):
    return await asyncio.wait_for(ch.recv_message(), timeout=timeout)


async def _collect_until_result(ch: PipeChannel, already=None) -> tuple[list, P.NodeResult]:
    """收消息直到拿到 NodeResult，返回 (progress 列表, NodeResult)。"""
    progresses = list(already or [])
    progresses = [m for m in progresses if isinstance(m, P.NodeProgress)]
    while True:
        msg = await _recv(ch)
        if isinstance(msg, P.NodeResult):
            return progresses, msg
        if isinstance(msg, P.NodeProgress):
            progresses.append(msg)


async def _shutdown(proc, ch: PipeChannel):
    ch.close()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3.0)


@pytest.mark.asyncio
async def test_run_tts_node_resolves_with_audio_result():
    """LoadModel(fake-tts) -> RunNode(node_type=tts) -> NodeResult(completed, audio outputs)。"""
    proc, ch = _spawn_tts_runner()
    try:
        await _recv(ch)  # 吞掉 Ready
        await ch.send_message(P.LoadModel(model_key="fake-tts-a", config={}))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent) and ev.event == "loaded"

        await ch.send_message(P.RunNode(
            task_id=21, node_id="tts", node_type="tts",
            model_key="fake-tts-a",
            inputs={"text": "你好世界", "voice": "default", "sample_rate": 24000},
            is_deterministic=False,
        ))
        _, result = await _collect_until_result(ch)
        assert result.status == "completed"
        assert result.outputs is not None
        # TTS 结果带 audio 元数据（不是 image 的 path/meta 形状）
        assert result.outputs["media_type"] == "audio/wav"
        assert result.outputs["meta"]["format"] == "wav"
        assert result.outputs["meta"]["sample_rate"] == 24000
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_run_node_unknown_type_fails():
    """node_type 不是 image / tts -> NodeResult status=failed，不崩 runner。"""
    proc, ch = _spawn_tts_runner()
    try:
        await _recv(ch)  # Ready
        await ch.send_message(P.LoadModel(model_key="fake-tts-a", config={}))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent) and ev.event == "loaded"

        await ch.send_message(P.RunNode(
            task_id=22, node_id="weird", node_type="video",
            model_key="fake-tts-a", inputs={}, is_deterministic=False,
        ))
        _, result = await _collect_until_result(ch)
        assert result.status == "failed"
        assert "node_type" in (result.error or "")
        # runner 仍活着 —— 再发一个 ping
        await ch.send_message(P.Ping())
        pong = await _recv(ch)
        assert isinstance(pong, P.Pong)
        assert pong.runner_id
    finally:
        await _shutdown(proc, ch)


@pytest.mark.asyncio
async def test_tts_node_abort_at_boundary():
    """spec 4.4：TTS = boundary-cancel only。dispatch 前 Abort -> NodeResult cancelled。"""
    proc, ch = _spawn_tts_runner()
    try:
        await _recv(ch)  # Ready
        await ch.send_message(P.LoadModel(model_key="fake-tts-a", config={}))
        ev = await _recv(ch)
        assert isinstance(ev, P.ModelEvent) and ev.event == "loaded"
        # 先 Abort 再 RunNode —— pipe-reader 把 Abort 落进 pending_aborts，
        # 下一条 RunNode 到达时 cancel_flag 立即置位，node-executor 在 dispatch
        # 前 boundary check 看到 → CancelledError → cancelled。覆盖「客户端要
        # 取消的节点还没被 runner 拿到」时序。
        await ch.send_message(P.Abort(task_id=23, node_id="tts"))
        await ch.send_message(P.RunNode(
            task_id=23, node_id="tts", node_type="tts",
            model_key="fake-tts-a", inputs={"text": "长文本" * 20},
            is_deterministic=False,
        ))
        _, result = await _collect_until_result(ch)
        assert result.status == "cancelled"
        assert result.task_id == 23
    finally:
        await _shutdown(proc, ch)
