# V1.5 Lane F: TTS runner 迁入 Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: 把 TTS 推理路径迁进 image/TTS runner 子进程，复用 Lane D 为 image runner
建立的「真 ModelManager 进 runner + per-model lock + OOM 处理」模式。Lane F 是
V1.5 最小的 Lane —— 重活 Lane D 已经做完（ModelManager-into-runner、per-model
`asyncio.Lock`、OOM evict 重试）。Lane F 只做三件事：

1. 把 Lane C / Lane D 的 node-executor 从「写死 `ImageRequest`」泛化成按
   `RunNode.node_type` 分流 —— `node_type == "tts"` 时构造 `AudioRequest`。
2. 让 TTS runner 走真 `TTSEngine` adapter（`src.workers.tts_engines.*`），通过
   真 `ModelManager`（Lane D 迁入的那个实例）`get_loaded_adapter` 取 adapter。
3. 验证 TTS 专属行为：spec §4.4 明确 TTS 节点只需 boundary-cancel（节点边界
   check），不需要 image sampler 那种 within-node（每 step）cancel —— TTS
   adapter 的 `infer(req)` 不接 `progress_callback` / `cancel_flag` kwargs。

Architecture: 三块改动，全部在已有文件上叠加，不新建生产模块。

1. node-executor 泛化（`src/runner/runner_process.py`）—— 把 Lane C/D 里写死
   `ImageRequest(...)` 的那段抽成 `_build_request(node)`：`node_type == "image"`
   → `ImageRequest`；`node_type == "tts"` → `AudioRequest`；其余 → `NodeResult
   status=failed`。TTS 分支调 `adapter.infer(req)`（不传 progress/cancel
   kwargs，因为 `TTSEngine.infer` 签名只收 `req` —— boundary-cancel only）。
2. TTS runner 配置（`configs/hardware.yaml` 注释 + runner spawn 入参）—— Lane A
   的 `hardware.3gpu.yaml` 已有 `role: tts` group；Lane F 不改 yaml schema，只
   确认 supervisor 对 `role` ∈ {image, tts} 的 group 都 spawn image/TTS runner
   （同一 `runner_main`，`adapter_class` 由 ModelManager 按 spec 决定）。
3. TTS 节点改道（`src/services/nodes/audio.py`）—— `TTSEngineNode.invoke` 当前
   直接 `we._model_manager.get_loaded_adapter(engine_name)` 在主进程内跑
   adapter；Lane S 把 workflow_executor 改成 dispatch 节点到 runner，Lane F 只
   需确认 TTS 节点被归类为 dispatch 节点（`node_type == "tts"`），实际改道由
   Lane S 完成。Lane F 在此只加一个回归测试锁住 `tts_engine` → dispatch 判定。

Tech Stack: Python 3.12 / `multiprocessing`（spawn context）/ `asyncio` / pytest
（`asyncio_mode = "auto"`）/ Lane C 的 `FakeAdapter` + 新增 `FakeTTSAdapter`
（零 GPU 的 `TTSEngine` 形状）做 runner 子进程测试，不碰真 GPU / 真 TTS 权重。

> 注意 — 与简报 / spec 的偏差（已核实，须知会）：
>
> 1. Lane D 的实施计划文件 `docs/superpowers/plans/2026-05-14-v15-laneD-modelmanager-into-runner.md`
>    在本仓库 plans 目录中不存在（只有 Lane 0/A/B/C/G/S）。简报说「Lane F 跟随
>    Lane D 的模式」—— 本计划据 spec §2.2「image/TTS Runner 内部」+ §4.3 OOM +
>    Lane C 的 `runner_process.py` 骨架推断 Lane D 的交付形状：Lane D 把
>    `_RunnerState.adapters` 那个极简 dict 换成真 `ModelManager` 实例，把
>    `_node_executor` 里取 adapter 的路径换成 `model_manager.get_loaded_adapter`。
>    Lane F 在此基础上加 TTS 分支。如 Lane D 实际交付形状不同，执行 Lane F 前
>    须先核对 Lane D 的 `runner_process.py` 改动，再调整本计划 Task 2 的
>    `old_string` 锚点。已在 Self-Review 标注为最大不确定点。
>
> 2. Lane C 的 `_node_executor`（`runner_process.py`）当前写死
>    `ImageRequest(request_id=..., prompt=..., steps=...)` 并调
>    `adapter.infer(req, progress_callback=..., cancel_flag=...)`。这是唯一
>    image-specific 的点。Lane F 把它泛化成 `_build_request(node)` 按 node_type
>    分流。TTS 分支不传 `progress_callback` / `cancel_flag` —— 真
>    `TTSEngine.infer(req)`（`src/workers/tts_engines/base.py:64`）签名只收
>    `req`，spec §4.4 也明确 TTS = boundary-cancel only。
>
> 3. 真 TTS adapter 已经是合规的 `InferenceAdapter` 子类。`TTSEngine`
>    （`base.py:32`）继承 `InferenceAdapter`，`load` 已 `asyncio.to_thread`
>    包装，`infer` 已 `asyncio.to_thread(self.synthesize)` 包装 —— spec §4.4
>    要求的「adapter.run 须为 async 可让出」TTS 侧天然满足，无需 Lane G 那种
>    adapter 重写。Lane F 不改任何 `src/workers/tts_engines/*.py`。
>
> 4. `configs/models.yaml` 已有 6 个 TTS adapter 条目（cosyvoice2 / indextts2 /
>    moss_tts / qwen3_tts_base / qwen3_tts_customvoice / qwen3_tts_voicedesign /
>    voxcpm2），`type: tts`，`adapter` 是 dotted path。Lane F 不新增 yaml 条目，
>    只确认这些条目能被 runner 内 ModelManager 的 `_instantiate_adapter` 正确
>    实例化（已 grep 确认 `model_manager.py:63` 的 dotted-path 解析对 TTS 通用）。

---

## File Structure

| 文件 | Lane F 动作 | 责任 |
|---|---|---|
| `backend/src/runner/runner_process.py` | 修改 | `_node_executor` 取 adapter + 构造 request 泛化为按 `node_type` 分流；新增 `_build_request(node)`；TTS 分支调 `adapter.infer(req)` 不传 progress/cancel kwargs |
| `backend/src/runner/fake_tts_adapter.py` | 新建 | `FakeTTSAdapter(TTSEngine)`：零 GPU 的 TTS adapter，`synthesize` 返回固定 wav bytes，给 runner 子进程 TTS 路径测试用 |
| `backend/tests/test_runner_tts_node.py` | 新建 | 真 `multiprocessing.Process` 跑 runner：LoadModel(fake-tts) → RunNode(node_type=tts) → NodeResult(audio outputs)；node_type 未知 → failed；TTS 节点 Abort 在边界生效 |
| `backend/tests/test_tts_node_is_dispatch.py` | 新建 | 回归：`tts_engine` 节点被归类为 dispatch 节点（`node_type == "tts"`），与 llm 节点（inline-HTTP）区分 |

> 测试基础设施复用：`tests/conftest.py` 强制 `ADMIN_PASSWORD=""` + `NOUS_DISABLE_BG_TASKS=1` + `CUDA_VISIBLE_DEVICES=""`。Lane F 的 runner 子进程测试不碰 app / DB；`FakeTTSAdapter` 继承 `TTSEngine` 但 `load_sync` / `synthesize` 不 import torch，所以 conftest 的 CUDA 隐藏不影响它。

---

## Task 1: `FakeTTSAdapter` —— 零 GPU 的 TTSEngine 实现

Lane C 的 `FakeAdapter` 是 `MediaModality.IMAGE`、`infer` 收 `progress_callback` /
`cancel_flag` kwargs，对应 image adapter 的形状。TTS adapter 形状不同：
`TTSEngine.infer(req)` 只收 `req`（boundary-cancel only），`synthesize` 是 sync
blocking。Lane F 需要一个零 GPU 的 `TTSEngine` 子类做 runner 子进程 TTS 路径
测试 —— `FakeTTSAdapter`。它实现 `load_sync` / `synthesize`，返回固定的最小 wav
bytes，不 import torch。

Files:
- Create: `backend/src/runner/fake_tts_adapter.py`
- Test: `backend/tests/test_runner_tts_node.py`（新建，本 Task 只先写 adapter 单测部分）

- [ ] Step 1: 写失败测试 — FakeTTSAdapter 是合规 TTSEngine

新建 `backend/tests/test_runner_tts_node.py`（本 Task 先放 adapter 单测，Task 2 再追加子进程测试）：
```python
"""Lane F: TTS runner 迁入测试 —— FakeTTSAdapter 单测 + runner 子进程 TTS 路径。

零 GPU / 零真模型：FakeTTSAdapter 继承 TTSEngine 但 synthesize 返回固定 wav
bytes，不 import torch。runner 子进程测试起真 multiprocessing.Process。
"""
import asyncio
import uuid

import pytest

from src.runner.fake_tts_adapter import FakeTTSAdapter
from src.services.inference.base import (
    AudioRequest,
    InferenceAdapter,
    InferenceResult,
    MediaModality,
)
from src.workers.tts_engines.base import TTSEngine


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
```

- [ ] Step 2: 跑测试确认失败

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_tts_node.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.runner.fake_tts_adapter'`。

- [ ] Step 3: 实现 `fake_tts_adapter.py`

新建 `backend/src/runner/fake_tts_adapter.py`：
```python
"""FakeTTSAdapter —— 零 GPU / 零真模型的 TTSEngine 实现。

V1.5 Lane F 用它跑通 TTS runner 子进程路径（IPC + 生命周期 + node_type=tts
分流）而不需要真硬件 / 真 TTS 权重。继承 src.workers.tts_engines.base.TTSEngine
（即 InferenceAdapter 的 TTS 子类），所以 runner / ModelManager 看它和真 TTS
adapter 形状完全一致。

不 import torch / torchaudio / soundfile —— synthesize 直接拼一段最小合法 WAV
header + 静音 PCM，conftest 的 CUDA_VISIBLE_DEVICES="" 对它无影响。

spec §4.4：TTS 节点只需 boundary-cancel。TTSEngine.infer(req) 签名只收 req，
不接 progress_callback / cancel_flag —— FakeTTSAdapter 保持这个形状。
"""
from __future__ import annotations

import struct
from typing import Any, ClassVar

from src.services.inference.base import MediaModality
from src.workers.tts_engines.base import TTSEngine, TTSResult


def _silent_wav(sample_rate: int, duration_seconds: float) -> bytes:
    """拼一段最小合法 WAV（PCM16 单声道静音）—— 不依赖 soundfile / torch。"""
    n_samples = max(1, int(sample_rate * duration_seconds))
    data = b"\x00\x00" * n_samples  # PCM16 静音
    byte_rate = sample_rate * 2
    block_align = 2
    riff_chunk_size = 36 + len(data)
    header = b"RIFF" + struct.pack("<I", riff_chunk_size) + b"WAVE"
    fmt = (
        b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
    )
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt + data_chunk


class FakeTTSAdapter(TTSEngine):
    """假 TTS adapter：load 无副作用，synthesize 返回固定静音 WAV。"""

    ENGINE_NAME: ClassVar[str] = "fake_tts"
    estimated_vram_mb: ClassVar[int] = 0
    modality = MediaModality.AUDIO

    def __init__(self, paths: dict[str, str], device: str = "cpu", **params: Any) -> None:
        super().__init__(paths=paths, device=device, **params)
        # fail_load 开关 —— 模拟权重丢失 / OOM，供 runner OOM/load-failed 路径测试
        self._fail_load = bool(params.get("fail_load", False))

    def load_sync(self) -> None:
        if self._fail_load:
            raise RuntimeError(f"fake tts load failure for paths={self.paths}")
        self._model = object()  # 非 None → is_loaded True

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("FakeTTSAdapter not loaded. Call load() first.")
        # 文本越长「时长」越长 —— 给 audio_seconds 一个可断言的非零值
        duration = round(max(0.1, len(text) * 0.05), 2)
        return TTSResult(
            audio_bytes=_silent_wav(sample_rate, duration),
            sample_rate=sample_rate,
            duration_seconds=duration,
            format="wav",
        )

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
```

- [ ] Step 4: 跑测试确认通过

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_tts_node.py -v`
Expected: 4 个用例全 PASS（`test_fake_tts_adapter_*`）。

- [ ] Step 5: lint 预检 + Commit

```bash
cd backend && ruff check src/runner/fake_tts_adapter.py tests/test_runner_tts_node.py
git add src/runner/fake_tts_adapter.py tests/test_runner_tts_node.py
git commit -m "feat(runner): add FakeTTSAdapter for GPU-free TTS runner tests

FakeTTSAdapter subclasses TTSEngine (the InferenceAdapter TTS subclass)
with no torch/torchaudio/soundfile — synthesize() returns a minimal
silent WAV. Mirrors Lane C's FakeAdapter pattern for the TTS modality
so the TTS runner path can be tested with no real GPU or TTS weights.
V1.5 Lane F, spec 4.4."
```

---

## Task 2: node-executor 按 node_type 分流（`runner_process.py`）

Lane C / Lane D 的 `_node_executor` 写死 `ImageRequest(...)` 并调
`adapter.infer(req, progress_callback=..., cancel_flag=...)`。Lane F 把「构造
request」抽成 `_build_request(node)` 按 `node_type` 分流，并让 TTS 分支调
`adapter.infer(req)`（不传 progress/cancel kwargs —— spec §4.4 TTS =
boundary-cancel only，真 `TTSEngine.infer(req)` 签名也只收 req）。

> 执行前提：本 Task 的 `old_string` 锚点基于 Lane C 的 `runner_process.py`
> Step 3 实现（见 Lane C 计划 1246-1310 行）。如 Lane D 已修改 `_node_executor`
> （把极简 adapter dict 换成真 ModelManager），须先 `Read` 当前
> `runner_process.py`，按 Lane D 实际改动调整下面的 `old_string`。语义不变：
> 把「写死 ImageRequest + image 专属 kwargs」改成「按 node_type 分流」。

Files:
- Modify: `backend/src/runner/runner_process.py`
- Test: `backend/tests/test_runner_tts_node.py`（追加子进程测试）

- [ ] Step 1: 写失败测试 — runner 子进程跑 TTS 节点

在 `backend/tests/test_runner_tts_node.py` 末尾追加：
```python
# ---- runner 子进程 TTS 路径（真 multiprocessing.Process）----

import multiprocessing as mp

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.supervisor import RunnerSupervisor


@pytest.fixture
def tts_supervisor():
    """起一个 fake TTS runner 子进程（adapter_class=FakeTTSAdapter）。"""
    ctx = mp.get_context("spawn")
    sup = RunnerSupervisor(
        group_id="tts",
        gpus=[3],
        ctx=ctx,
        adapter_class="src.runner.fake_tts_adapter.FakeTTSAdapter",
    )
    yield sup
    asyncio.get_event_loop().run_until_complete(sup.stop())


@pytest.mark.asyncio
async def test_run_tts_node_resolves_with_audio_result():
    """LoadModel(fake-tts) -> RunNode(node_type=tts) -> NodeResult(completed, audio outputs)。"""
    ctx = mp.get_context("spawn")
    sup = RunnerSupervisor(
        group_id="tts", gpus=[3], ctx=ctx,
        adapter_class="src.runner.fake_tts_adapter.FakeTTSAdapter",
    )
    await sup.start()
    try:
        client: RunnerClient = sup.client
        await client.load_model("fake_tts", config={})
        result = await client.run_node(P.RunNode(
            task_id=21, node_id="tts", node_type="tts",
            model_key="fake_tts",
            inputs={"text": "你好世界", "voice": "default", "sample_rate": 24000},
            is_deterministic=False,
        ))
        assert result.status == "completed"
        assert result.outputs is not None
        # TTS 结果带 audio 元数据（不是 image 的 path/meta 形状）
        assert result.outputs["media_type"] == "audio/wav"
        assert result.outputs["meta"]["format"] == "wav"
        assert result.outputs["meta"]["sample_rate"] == 24000
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_run_node_unknown_type_fails():
    """node_type 不是 image / tts -> NodeResult status=failed，不崩 runner。"""
    ctx = mp.get_context("spawn")
    sup = RunnerSupervisor(
        group_id="tts", gpus=[3], ctx=ctx,
        adapter_class="src.runner.fake_tts_adapter.FakeTTSAdapter",
    )
    await sup.start()
    try:
        client: RunnerClient = sup.client
        await client.load_model("fake_tts", config={})
        result = await client.run_node(P.RunNode(
            task_id=22, node_id="weird", node_type="video",
            model_key="fake_tts", inputs={}, is_deterministic=False,
        ))
        assert result.status == "failed"
        assert "node_type" in (result.error or "")
        # runner 仍活着 —— 再发一个 ping
        pong = await client.ping()
        assert pong.runner_id
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_tts_node_abort_at_boundary():
    """spec 4.4：TTS = boundary-cancel only。dispatch 前 Abort -> NodeResult cancelled。"""
    ctx = mp.get_context("spawn")
    sup = RunnerSupervisor(
        group_id="tts", gpus=[3], ctx=ctx,
        adapter_class="src.runner.fake_tts_adapter.FakeTTSAdapter",
    )
    await sup.start()
    try:
        client: RunnerClient = sup.client
        await client.load_model("fake_tts", config={})
        # 先 Abort 再 RunNode —— 节点边界 check 应直接判 cancelled
        await client.send(P.Abort(task_id=23, node_id="tts"))
        result = await client.run_node(P.RunNode(
            task_id=23, node_id="tts", node_type="tts",
            model_key="fake_tts", inputs={"text": "长文本" * 20},
            is_deterministic=False,
        ))
        assert result.status == "cancelled"
    finally:
        await sup.stop()
```

> 测试说明：`RunnerSupervisor` / `RunnerClient` 的构造与方法名（`start` /
> `stop` / `client` / `load_model` / `run_node` / `ping` / `send`）以 Lane C
> 实际交付为准。如 Lane C 命名不同，执行前按 Lane C 的 `test_runner_supervisor.py`
> / `test_runner_client.py` 对齐 —— Lane F 不改 supervisor / client。

- [ ] Step 2: 跑测试确认失败

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_tts_node.py -k "tts_node or unknown_type" -v`
Expected: FAIL —— `_node_executor` 写死 `ImageRequest`，`node_type="tts"` 时
要么构造出错的 request、要么 `adapter.infer` 收到非 AudioRequest 抛 TypeError，
`test_run_tts_node_resolves_with_audio_result` 拿不到 `audio/wav` 结果；
`test_run_node_unknown_type_fails` 也不会有 `"node_type"` 错误信息。

- [ ] Step 3: 实现 — 抽 `_build_request` + node_type 分流

编辑 `backend/src/runner/runner_process.py`。

(a) 在 `_node_executor` 之前新增 `_build_request` helper：
```python
def _build_request(node: P.RunNode):
    """按 node_type 构造 typed InferenceRequest。

    spec §3.3：RunNode.node_type 仅 "image" / "tts"。image 走 ImageRequest
    （within-node cancel，progress callback per step）；tts 走 AudioRequest
    （spec §4.4：boundary-cancel only，infer(req) 不接 progress/cancel kwargs）。
    未知 node_type 抛 ValueError —— node-executor 转成 NodeResult status=failed。
    """
    from src.services.inference.base import AudioRequest, ImageRequest

    if node.node_type == "image":
        return ImageRequest(
            request_id=f"task-{node.task_id}",
            prompt=str(node.inputs.get("prompt", "")),
            negative_prompt=str(node.inputs.get("negative_prompt", "")),
            steps=int(node.inputs.get("steps", 1) or 1),
        )
    if node.node_type == "tts":
        return AudioRequest(
            request_id=f"task-{node.task_id}",
            text=str(node.inputs.get("text", "")),
            voice=str(node.inputs.get("voice", "default")),
            speed=float(node.inputs.get("speed", 1.0) or 1.0),
            sample_rate=int(node.inputs.get("sample_rate", 24000) or 24000),
        )
    raise ValueError(f"unsupported node_type {node.node_type!r} (expected image / tts)")
```

(b) 把 `_node_executor` 里写死 `ImageRequest` + 调 `adapter.infer` 的那段替换。
找到 Lane C 实现的这段：
```python
        try:
            from src.services.inference.base import ImageRequest

            req = ImageRequest(
                request_id=f"task-{node.task_id}",
                prompt=str(node.inputs.get("prompt", "")),
                steps=int(node.inputs.get("steps", 1) or 1),
            )
            result = await adapter.infer(
                req, progress_callback=_on_progress, cancel_flag=cancel_flag,
            )
```
替换成：
```python
        try:
            req = _build_request(node)
            if node.node_type == "tts":
                # spec §4.4：TTS = boundary-cancel only。TTSEngine.infer(req)
                # 签名只收 req —— 不传 progress_callback / cancel_flag。节点边界
                # 的 cancel 由 pipe-reader 在 dispatch 前置位的 cancel_flag +
                # 下面的 boundary check 覆盖。
                if cancel_flag.is_set():
                    raise asyncio.CancelledError()
                result = await adapter.infer(req)
            else:
                # image：within-node cancel + per-step progress（Lane G adapter 重写）
                result = await adapter.infer(
                    req, progress_callback=_on_progress, cancel_flag=cancel_flag,
                )
        except ValueError as e:
            # 未知 node_type —— 明确判 failed，不崩 runner
            await ch.send_message(P.NodeResult(
                task_id=node.task_id, node_id=node.node_id, status="failed",
                outputs=None, error=str(e),
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            state.cancel_flags.pop(node.task_id, None)
            continue
```

> 注意：`except ValueError` 块要放在 Lane C 已有的 `except asyncio.CancelledError`
> / `except Exception` 之前 —— `ValueError` 是 `Exception` 子类，顺序错了会被
> 泛 catch 吞掉、错误信息里就没有 `_build_request` 的 `"node_type"` 字样，
> `test_run_node_unknown_type_fails` 会失败。Lane C 已有的 cancelled / failed
> 分支不动。

(c) Lane C 已有的 `NodeResult` completed 分支用 `result.metadata` /
`result.media_type` 拼 outputs —— TTS 的 `InferenceResult.media_type` 是
`audio/wav`、`metadata` 带 `sample_rate` / `format`，这段对 TTS 天然通用，不改。

- [ ] Step 4: 跑测试确认通过

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_tts_node.py -v`
Expected: 全部 PASS（4 个 adapter 单测 + 3 个子进程测试）。子进程测试起真
`multiprocessing.Process`，单文件约 15-25s。

回归确认 image 路径没被改坏：
Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_runner_process.py -v`
Expected: Lane C 的 5 个用例仍全 PASS（`node_type="image"` 走 `_build_request`
的 image 分支，行为与 Lane C 写死时一致）。

- [ ] Step 5: lint 预检 + Commit

```bash
cd backend && ruff check src/runner/runner_process.py tests/test_runner_tts_node.py
git add src/runner/runner_process.py tests/test_runner_tts_node.py
git commit -m "feat(runner): dispatch node-executor by node_type for TTS migration

Generalize _node_executor's hardcoded ImageRequest into _build_request(node)
that branches on RunNode.node_type: image -> ImageRequest (within-node
cancel + per-step progress), tts -> AudioRequest. TTS calls adapter.infer(req)
with no progress/cancel kwargs — spec 4.4 makes TTS boundary-cancel only,
and TTSEngine.infer(req) takes only req. Unknown node_type -> NodeResult
failed without crashing the runner. V1.5 Lane F, spec 3.3 / 4.4."
```

---

## Task 3: 回归锁 — TTS 节点归类为 dispatch 节点

spec §2.2「节点分流阶段」+ §4.5 D6 改道清单：image/TTS 节点是 dispatch 节点
（投到 runner 串行队列），llm 节点是 inline-HTTP 节点（主进程直连 vLLM）。实际
分流逻辑由 Lane S 的 workflow_executor 重写实现。Lane F 在此只加一个回归测试，
锁住「`tts_engine` 节点 → dispatch / `node_type == "tts"`」这个判定 —— 防止后续
Lane 把 TTS 误归类成 inline-HTTP 节点（那会绕过 runner 串行队列、重新引入
spec §1.3 要解决的并发竞态）。

> 执行前提：本 Task 的判定函数（spec 草图叫 `is_dispatch_node`）以 Lane S 实际
> 交付为准。如执行 Lane F 时 Lane S 尚未合并，本 Task 改为对 ModelManager 的
> `model_type` 映射做回归（`configs/models.yaml` 里 TTS 条目 `type: tts`，已
> 确认）—— 见下面 Step 1 的两种写法，按 Lane S 是否已合并二选一。

Files:
- Test: `backend/tests/test_tts_node_is_dispatch.py`（新建）

- [ ] Step 1: 写回归测试

新建 `backend/tests/test_tts_node_is_dispatch.py`：
```python
"""Lane F 回归：TTS 节点必须被归类为 dispatch 节点（进 runner 串行队列）。

spec §1.3 要解决的并发竞态正是靠「image/TTS 节点进 per-group 串行队列」根治。
若后续 Lane 把 tts_engine 误归类成 inline-HTTP 节点，会绕过串行队列、重新引入
adapter race —— 本测试锁死这个判定。

写法 A（Lane S 已合并）：直接断言 is_dispatch_node。
写法 B（Lane S 未合并）：退而断言 models.yaml 里 TTS 条目 model_type == "tts"，
                        且 audio 节点 registry 注册了 tts_engine。
"""
import pytest


def test_tts_engine_node_is_dispatch_node():
    """写法 A：Lane S 的 is_dispatch_node 把 tts_engine 判为 dispatch。"""
    try:
        from src.services.workflow_executor import is_dispatch_node
    except ImportError:
        pytest.skip("Lane S workflow_executor rewrite not merged yet — see 写法 B")
    assert is_dispatch_node({"type": "tts_engine"}) is True
    assert is_dispatch_node({"type": "image_diffusers"}) is True
    # llm 节点是 inline-HTTP，不是 dispatch
    assert is_dispatch_node({"type": "llm"}) is False


def test_tts_model_specs_typed_tts():
    """写法 B：configs/models.yaml 里 TTS adapter 条目 model_type == 'tts'。

    runner 内 ModelManager 按 spec.model_type 决定 adapter 形状；TTS 条目必须
    typed 'tts' 才会落进 TTS runner group。这条永远跑（不依赖 Lane S）。
    """
    from src.services.inference.registry import ModelRegistry

    reg = ModelRegistry("configs/models.yaml")
    tts_specs = reg.list_by_type("tts")
    assert tts_specs, "configs/models.yaml 应至少有一个 type: tts 的 adapter 条目"
    for spec in tts_specs:
        # adapter_class 是可解析的 dotted path（runner 内 ModelManager 要 import 它）
        assert "." in spec.adapter_class
        assert spec.adapter_class.startswith("src.workers.tts_engines.")


def test_tts_engine_node_registered():
    """写法 B 续：audio 节点 registry 注册了 tts_engine handler。"""
    from src.services.nodes.registry import get_node_handler

    handler = get_node_handler("tts_engine")
    assert handler is not None
```

- [ ] Step 2: 跑测试确认状态

Run: `cd backend && ADMIN_PASSWORD="" python -m pytest tests/test_tts_node_is_dispatch.py -v`
Expected: `test_tts_model_specs_typed_tts` + `test_tts_engine_node_registered`
PASS（不依赖 Lane S）；`test_tts_engine_node_is_dispatch_node` 视 Lane S 是否
合并 —— 已合并则 PASS，未合并则 SKIP（`pytest.skip`）。三者都不应 FAIL。

> 若 `get_node_handler` 在 `src/services/nodes/registry.py` 里的实际名字不同
> （Lane F 不改该文件），按实际 API 调整 `test_tts_engine_node_registered` 的
> import —— grep `def .*node` `src/services/nodes/registry.py` 确认。

- [ ] Step 3: lint 预检 + Commit

```bash
cd backend && ruff check tests/test_tts_node_is_dispatch.py
git add tests/test_tts_node_is_dispatch.py
git commit -m "test(runner): lock TTS node as dispatch node (regression)

TTS nodes must route to the runner serial queue, not inline-HTTP — that
serial queue is what kills the spec 1.3 adapter-race. Regression test
asserts is_dispatch_node({type: tts_engine}) (when Lane S merged) and,
unconditionally, that models.yaml TTS specs are typed 'tts' with
resolvable adapter dotted paths. V1.5 Lane F, spec 1.3 / 2.2."
```

---

## 最终验证 + PR

- [ ] 跑 Lane F 全部测试 + runner 框架回归：
```bash
cd backend && ADMIN_PASSWORD="" python -m pytest \
  tests/test_runner_tts_node.py \
  tests/test_tts_node_is_dispatch.py \
  tests/test_runner_process.py \
  tests/test_fake_adapter.py \
  -v
```
Expected: Lane F 新增用例全 PASS（或 Lane S 未合并时 1 个 SKIP）；Lane C 的
`test_runner_process.py` / `test_fake_adapter.py` 仍全 PASS（node_type 分流没
改坏 image 路径）。

- [ ] 全量 lint 预检：
```bash
cd backend && ruff check src/runner/ tests/test_runner_tts_node.py tests/test_tts_node_is_dispatch.py
```
Expected: `All checks passed!`

- [ ] 推分支 + 开 PR：
```bash
git push -u origin spec/v15-laneF-tts-runner
gh pr create --title "feat: V1.5 Lane F — TTS runner 迁入" --body "$(cat <<'EOF'
## Summary
- 把 TTS 推理路径迁进 image/TTS runner 子进程，复用 Lane D 的 ModelManager-into-runner 模式
- node-executor 从写死 ImageRequest 泛化为按 RunNode.node_type 分流（image / tts）
- TTS 走 boundary-cancel only（spec §4.4）—— adapter.infer(req) 不接 progress/cancel kwargs
- 新增 FakeTTSAdapter（零 GPU 的 TTSEngine）做 runner 子进程 TTS 路径测试
- 回归锁住 TTS 节点 = dispatch 节点（防止误归类绕过 runner 串行队列）

## Spec 覆盖
Lane F 在 spec「实施分 Lane」表的职责是「TTS runner 迁入。依赖：D」。Lane D 已
完成 ModelManager-into-runner / per-model lock / OOM —— Lane F 把该模式应用到
TTS adapter（已是合规 InferenceAdapter，无需 adapter 重写），并验证 spec §4.4
的 TTS = boundary-cancel only。

## Test plan
- [ ] `pytest tests/test_runner_tts_node.py -v` —— FakeTTSAdapter 单测 + runner 子进程 TTS 路径
- [ ] `pytest tests/test_tts_node_is_dispatch.py -v` —— TTS 节点 dispatch 归类回归
- [ ] `pytest tests/test_runner_process.py -v` —— Lane C image 路径回归（node_type 分流没改坏）
- [ ] `ruff check src/runner/` —— lint 绿
- [ ] CI 绿后 merge
EOF
)"
```

---

## Self-Review

**与 spec 的对应：** Lane F 对应 spec「实施分 Lane」表的「F：TTS runner 迁入。
依赖：D」。spec §1.2/§1.3 把 image/TTS runner 归为同一类（per-group 串行
PriorityQueue + per-model lock），spec 也明说 image/TTS runner 共用 `runner_main`
—— 所以 Lane F 不新建 runner 进程，只把 Lane C/D 的 node-executor 从 image-only
泛化成 image+tts。spec §4.4 明确「TTS / VAE / 短节点只在边界 check」—— Task 2 的
TTS 分支不传 `progress_callback` / `cancel_flag`、只在 dispatch 前做 boundary
check，与之一致。

**Lane D 依赖的处理：** 这是本计划最大的不确定点。简报和 spec Lane 表都说
Lane F 依赖 Lane D，且简报给了 Lane D 的计划文件路径 ——但该文件在仓库 plans
目录中不存在（只有 Lane 0/A/B/C/G/S）。本计划据 spec §2.2/§4.3 + Lane C 的
`runner_process.py` 骨架推断 Lane D 的交付：把 `_RunnerState.adapters` 极简 dict
换成真 `ModelManager`、把取 adapter 改成 `get_loaded_adapter`。Task 2 的
`old_string` 锚点基于 Lane C 的实现 —— 如 Lane D 已改写 `_node_executor`，执行
Lane F 前必须先 `Read` 当前 `runner_process.py` 重新对齐锚点。语义不变（image-only
→ node_type 分流），但行号 / 上下文可能漂移。已在计划头 + Task 2 前置提醒标注。

**为什么 Lane F 这么短（3 个 Task）：** 重活 Lane C + Lane D 已做完 —— IPC 协议、
pipe 桥接、双 task、supervisor、ModelManager-into-runner、per-model lock、OOM
都是通用的。真 TTS adapter（`TTSEngine`）本就是合规 `InferenceAdapter`、`load` /
`infer` 已 `asyncio.to_thread` 包装，不需要 Lane G 那种 adapter 重写。Lane F 唯一
的 image-specific 改动点是 node-executor 写死的 `ImageRequest` —— 泛化它 + 加
TTS 形状的 fake adapter + 加回归锁，就是全部。没有为凑长度而加的 Task。

**spec 歧义 / 偏差（须知会）：**
1. Lane D 计划文件不存在（上面已详述）—— 本计划据 spec 推断 Lane D 形状，执行
   前须核对。
2. spec §3.3 `RunNode.node_type` 注释写「仅 "image" / "tts"」，但 spec 正文多处
   写「TTS」大写、§4.4 标题写「image/TTS」—— Task 2 统一按 `node_type == "tts"`
   小写字面量（与 Lane C `protocol.py` 的 `RunNode` 注释一致）。
3. spec §2.2「节点分流阶段」的 `is_dispatch_node` 是草图函数名，实际由 Lane S
   交付。Task 3 用「写法 A / 写法 B」双轨：Lane S 已合并断言 `is_dispatch_node`，
   未合并则退到 `models.yaml` 的 `type: tts` 回归 —— 后者不依赖任何未合并 Lane，
   永远可跑。

**判断取舍：**
- Task 1 的 `FakeTTSAdapter` 选择继承真 `TTSEngine` 而非新写一个独立 fake ——
  这样 runner / ModelManager 看它和真 TTS adapter 形状完全一致（继承
  `TTSEngine.infer` 的 `isinstance(req, AudioRequest)` 校验、`engine_name`
  property 等），测试覆盖更接近真实路径。代价是 `FakeTTSAdapter` 要实现
  `load_sync` / `synthesize` 两个抽象方法 —— 用 struct 拼最小 WAV、不 import
  torch，零 GPU 成立。
- Task 2 把 `except ValueError` 放在 Lane C 已有的泛 `except Exception` 之前 ——
  必须如此，否则未知 node_type 的错误信息被泛 catch 吞掉、丢掉 `"node_type"`
  关键字，回归测试 `test_run_node_unknown_type_fails` 会失败。已在 Task 2 Step 3
  明确标注顺序要求。
- Task 3 不实际改 workflow_executor 的分流逻辑（那是 Lane S 的职责，spec Lane
  表 F 不依赖 S）—— Lane F 只加回归锁，把「TTS = dispatch 节点」这个不变量钉死，
  避免 Lane S / Lane I 后续误改。这是「Lane 边界纪律」：F 只碰 runner 侧 TTS
  迁入，不越界进 executor 重写。
- 没改任何 `src/workers/tts_engines/*.py` —— 真 TTS adapter 已合规，改它属于
  scope 蔓延。Lane F 全部改动集中在 `src/runner/` + 测试 + 一个回归锁。
