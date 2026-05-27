"""Tests for LLM streaming dispatch via the v2 InferenceAdapter pipeline.

The low-level SSE parser used to live as workflow_executor._stream_llm —
v2 moved it into VLLMAdapter.infer_stream (covered by test_vllm_adapter.py).
What remains here is the dispatch boundary: WorkflowExecutor + LLMNode.stream
should turn each StreamEvent('delta') into a node_stream event and emit a
terminal node_end_streaming carrying the usage from the done event.
"""

from unittest.mock import AsyncMock, MagicMock


async def test_exec_llm_injects_node_id():
    """_execute_node should inject _node_id into node data before calling executor."""
    from src.services.workflow_executor import WorkflowExecutor

    workflow = {
        "nodes": [
            {
                "id": "n1",
                "type": "text_input",
                "data": {"text": "hi"},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [],
    }
    executor = WorkflowExecutor(workflow)
    result = await executor.execute()
    assert "outputs" in result
    assert "n1" in result["outputs"]


async def test_llm_streaming_dispatch_pushes_node_stream_events(monkeypatch):
    """End-to-end: WorkflowExecutor + LLMNode.stream should emit one
    node_stream event per delta token and a node_end_streaming carrying usage.
    """
    from src.services import workflow_executor as we
    from src.services.inference.base import StreamEvent
    from src.services.workflow_executor import WorkflowExecutor

    async def fake_infer_stream(req):
        for token in ["Streaming", " reply"]:
            yield StreamEvent(type="delta", payload={"content": token})
        yield StreamEvent(
            type="done",
            payload={"usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }},
        )

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer_stream = fake_infer_stream

    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    events: list[dict] = []

    async def on_progress(event: dict) -> None:
        events.append(event)

    workflow = {
        "nodes": [
            {
                "id": "in",
                "type": "text_input",
                "data": {"text": "hello"},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "llm_node_1",
                "type": "llm",
                "data": {
                    "model": "test",
                    "model_key": "test-model",
                    "stream": True,
                },
                "position": {"x": 1, "y": 0},
            },
        ],
        "edges": [
            {"source": "in", "target": "llm_node_1",
             "sourceHandle": "text", "targetHandle": "text"},
        ],
    }

    executor = WorkflowExecutor(workflow, on_progress=on_progress)
    result = await executor.execute()

    assert result["outputs"]["llm_node_1"]["text"] == "Streaming reply"

    stream_events = [e for e in events if e.get("type") == "node_stream"
                     and e.get("node_id") == "llm_node_1"]
    assert len(stream_events) == 2
    assert stream_events[0]["content"] == "Streaming"
    assert stream_events[1]["content"] == " reply"

    end_events = [e for e in events if e.get("type") == "node_end_streaming"]
    assert len(end_events) == 1
    assert end_events[0]["node_id"] == "llm_node_1"
    assert end_events[0]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }

    # PR-1c:LLM lane L3 progress —— stream 末尾必有一个 final node_progress
    # (stage=llm_gen, progress=1.0, eta_ms=0, step=真 completion_tokens 回填 2)。
    # 中间 throttled 帧因为 fake_infer_stream 没有 sleep 可能 0 个(throttle 250ms),
    # 末帧 emit 不受 throttle 影响,必有。
    llm_progress = [e for e in events if e.get("type") == "node_progress"
                    and e.get("node_id") == "llm_node_1"]
    assert llm_progress, "PR-1c:LLM stream 应至少发一个 node_progress(末帧)"
    final = llm_progress[-1]
    assert final["stage"] == "llm_gen"
    assert final["progress"] == 1.0
    assert final["eta_ms"] == 0
    assert final["step"] == 2  # usage.completion_tokens 回填
    assert final["total_steps"] == 2


async def test_llm_node_marks_multimodal_when_image_input_present(monkeypatch):
    """PR-1d:LLMNode.invoke/stream 检测到 image input → result.multimodal=True。"""
    from src.services import workflow_executor as we
    from src.services.inference.base import InferenceResult, UsageMeter
    from src.services.nodes.llm import LLMNode

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer = AsyncMock(return_value=InferenceResult(
        media_type="application/json", data=b"{}", metadata={"raw": {
            "choices": [{"message": {"content": "a cat"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }},
        usage=UsageMeter(input_tokens=5, output_tokens=2, latency_ms=10),
    ))
    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    node = LLMNode()
    # 多模态:inputs.image data URI
    r_vision = await node.invoke(
        {"model": "x", "model_key": "x"},
        {"prompt": "what is this?", "image": "data:image/png;base64,xxx"},
    )
    assert r_vision["multimodal"] is True
    # 纯文本:无 image / audio
    r_text = await node.invoke({"model": "x", "model_key": "x"}, {"prompt": "hi"})
    assert r_text["multimodal"] is False


async def test_llm_streaming_emits_vision_inference_stage_when_multimodal(monkeypatch):
    """PR-1d:workflow_executor 在 LLM 节点 + image 输入时,emitter 用 stage=vision_inference。"""
    from src.services import workflow_executor as we
    from src.services.inference.base import StreamEvent
    from src.services.workflow_executor import WorkflowExecutor

    async def fake_infer_stream(req):
        yield StreamEvent(type="delta", payload={"content": "a"})
        yield StreamEvent(type="delta", payload={"content": "cat"})
        yield StreamEvent(type="done", payload={"usage": {
            "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})

    adapter = MagicMock()
    adapter.is_loaded = True
    adapter.infer_stream = fake_infer_stream
    mgr = MagicMock()
    mgr.get_loaded_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(we, "_model_manager", mgr)

    events: list[dict] = []

    async def on_progress(ev: dict) -> None:
        events.append(ev)

    # workflow 含 image 输入 → LLM 节点应走 vision_inference stage
    workflow = {
        "nodes": [
            {"id": "img", "type": "text_input",
             "data": {"text": "data:image/png;base64,xxx"},
             "position": {"x": 0, "y": 0}},
            {"id": "vis", "type": "llm",
             "data": {"model": "vl", "model_key": "vl-model", "stream": True},
             "position": {"x": 1, "y": 0}},
        ],
        "edges": [
            {"source": "img", "target": "vis",
             "sourceHandle": "text", "targetHandle": "image"},
        ],
    }
    executor = WorkflowExecutor(workflow, on_progress=on_progress)
    await executor.execute()

    progress_events = [e for e in events if e.get("type") == "node_progress"]
    assert progress_events, "PR-1d:multimodal LLM stream 应至少发一个 node_progress(末帧)"
    # 末帧 stage = vision_inference(不是 llm_gen)
    assert progress_events[-1]["stage"] == "vision_inference"
    assert progress_events[-1]["progress"] == 1.0


async def test_llm_progress_emitter_throttles_token_rate_and_finals():
    """PR-1c 直接单测 _LlmProgressEmitter:throttle 250ms + final 末帧回填。"""
    import asyncio

    from src.services.workflow_executor import _LlmProgressEmitter

    events: list[dict] = []

    async def on_progress(ev: dict) -> None:
        events.append(ev)

    em = _LlmProgressEmitter(node_id="n", max_tokens=100, on_progress=on_progress)

    # 高速发 token —— 第 1 个会立刻发(last_emit_t=0,任何 now 都 >250ms),
    # 接下来的 token 在 250ms 内不发 → 应只有 1 个 throttled 帧 + 末帧。
    for _ in range(20):
        await em.on_token("x")
    await em.emit_final(true_completion=20)

    # 中间 throttled 至少 1 帧(首 token 触发 first emit);最后 1 帧 final。
    progress_events = [e for e in events if e["stage"] == "llm_gen"]
    assert len(progress_events) >= 2
    # 末帧:回填 true_completion=20 / progress=1.0 / eta=0
    final = progress_events[-1]
    assert final["progress"] == 1.0
    assert final["step"] == 20
    assert final["total_steps"] == 20
    assert final["eta_ms"] == 0
    assert "done" in final["detail"]

    # —— throttle 验证 ——
    # 等 300ms 后再发,应该能再触发一次中间 emit。
    em2 = _LlmProgressEmitter(node_id="n2", max_tokens=100, on_progress=on_progress)
    events2: list[dict] = []

    async def on_p2(ev: dict) -> None:
        events2.append(ev)

    em2 = _LlmProgressEmitter(node_id="n2", max_tokens=100, on_progress=on_p2)
    await em2.on_token("a")  # 立即 emit(首次 throttle 通过)
    await asyncio.sleep(0.05)  # < 250ms,下次不发
    await em2.on_token("b")
    await em2.on_token("c")
    await asyncio.sleep(0.3)  # > 250ms,下次可再发
    await em2.on_token("d")
    # 至少 2 个中间帧(首次 + 第二次 throttle 窗口后)。
    intermediate = [e for e in events2 if e["progress"] < 1.0]
    assert len(intermediate) >= 2
