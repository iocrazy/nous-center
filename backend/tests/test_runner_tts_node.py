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


@pytest.mark.asyncio
async def test_fake_tts_infer_emits_tts_synth_start_and_end(monkeypatch):
    """PR-1b:TTSEngine.infer 默认 emit `tts_synth` start(progress=0)+ end(progress=1.0)。
    fake synthesize 同步返回(<1ms),ticker 没机会 fire,所以只看 start/end 两帧。
    end 帧 step_latency_ms 是 int / eta_ms == 0 / detail 含 duration_seconds。"""
    events: list[dict] = []

    def on_p(step: int, total: int, **extras) -> None:
        events.append({"step": step, "total": total, **extras})

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    await a.infer(_audio_req("你好"), progress_callback=on_p)

    # 至少 2 帧(start + end);ticker 可能没机会 fire(fake synth <1ms)。
    assert len(events) >= 2
    start, end = events[0], events[-1]
    # start
    assert start["stage"] == "tts_synth"
    assert start["progress"] == 0.0
    assert start["step"] == 0
    # end
    assert end["stage"] == "tts_synth"
    assert end["progress"] == 1.0
    assert end["eta_ms"] == 0
    assert isinstance(end["step_latency_ms"], int)
    assert "done" in end["detail"]


@pytest.mark.asyncio
async def test_fake_tts_infer_boundary_cancel_raises():
    """PR-1b spec §4.4 升级:boundary cancel 仍然立刻 raise CancelledError,
    synthesize 不会跑(节省 GPU/CPU)。"""
    import threading

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    flag = threading.Event()
    flag.set()
    with pytest.raises(asyncio.CancelledError):
        await a.infer(_audio_req("any"), cancel_flag=flag)


@pytest.mark.asyncio
async def test_fake_tts_infer_ticker_fires_under_slow_synthesize(monkeypatch):
    """PR-1b ticker 集成:synthesize 慢(500ms)时,periodic ticker 应至少 fire 一次
    (默认 300ms 间隔)。覆盖「synthesize 走 to_thread + 主 loop 跑 ticker」并发路径,
    本测试是「假合成 + 真 asyncio.to_thread」的最小集成,补足真 TTS 引擎本地无 dep
    跑不了的 smoke 验证缺口(torchaudio/qwen_tts/voxcpm/librosa 都不在本 dev 环境)。"""
    import time

    events: list[dict] = []

    def on_p(step: int, total: int, **extras) -> None:
        events.append({"step": step, "total": total, **extras})

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")

    # 真 to_thread 跑 500ms 阻塞 —— 主 loop 同时跑 ticker(300ms)→ 应抓到 >=1 中间帧。
    orig_synth = a.synthesize

    def slow_synth(*args, **kwargs):
        time.sleep(0.5)
        return orig_synth(*args, **kwargs)

    monkeypatch.setattr(a, "synthesize", slow_synth)
    await a.infer(_audio_req("hello"), progress_callback=on_p)

    # start + 至少 1 ticker + end ≥ 3 帧。
    assert len(events) >= 3, f"ticker 应 fire 至少 1 次 (got {len(events)} events)"
    # 中间帧 progress 严格在 (0, 1) 区间(ticker 上限 0.95)。
    middle = events[1:-1]
    assert middle, "ticker 中间帧应存在"
    for e in middle:
        assert 0 < e["progress"] < 1.0, f"ticker 帧 progress 应 ∈ (0,1),实际:{e['progress']}"
        assert e["stage"] == "tts_synth"


@pytest.mark.asyncio
async def test_fake_tts_infer_legacy_callback_signature_backcompat():
    """PR-1b:老 callback 只接 (done, total) → _make_tts_emit TypeError 降级。"""
    calls: list[tuple[int, int]] = []

    def on_p(step: int, total: int) -> None:  # 老契约
        calls.append((step, total))

    a = FakeTTSAdapter(paths={"main": "/fake/tts"})
    await a.load("cpu")
    await a.infer(_audio_req("hi"), progress_callback=on_p)
    # start (0, est) + end (n, n)
    assert len(calls) >= 2
    assert calls[0][0] == 0
    assert calls[-1][0] == calls[-1][1]


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
