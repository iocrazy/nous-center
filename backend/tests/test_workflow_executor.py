from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.services.workflow_executor import WorkflowExecutor, ExecutionError


def _simple_workflow():
    """text_input -> output"""
    return {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "audio"},
        ],
    }


def _tts_workflow():
    """text_input -> tts_engine -> output"""
    return {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "你好"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "tts_engine", "data": {"engine": "cosyvoice2", "speed": 1.0, "voice": "default", "sample_rate": 24000}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "audio", "target": "n3", "targetHandle": "audio"},
        ],
    }


def test_topological_sort():
    wf = _simple_workflow()
    executor = WorkflowExecutor(wf)
    order = executor._topological_sort()
    assert order.index("n1") < order.index("n2")


def test_topological_sort_tts():
    wf = _tts_workflow()
    executor = WorkflowExecutor(wf)
    order = executor._topological_sort()
    assert order.index("n1") < order.index("n2")
    assert order.index("n2") < order.index("n3")


def test_cycle_detection():
    wf = {
        "nodes": [
            {"id": "a", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}},
            {"id": "b", "type": "output", "data": {}, "position": {"x": 0, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "sourceHandle": "text", "target": "b", "targetHandle": "audio"},
            {"id": "e2", "source": "b", "sourceHandle": "audio", "target": "a", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    with pytest.raises(ExecutionError, match="循环依赖"):
        executor._topological_sort()


def test_empty_workflow():
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    with pytest.raises(ExecutionError, match="空"):
        executor._topological_sort()


@pytest.mark.asyncio
async def test_execute_text_passthrough():
    wf = _simple_workflow()
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n1"]["text"] == "hello"


@pytest.mark.asyncio
async def test_get_inputs():
    wf = _tts_workflow()
    executor = WorkflowExecutor(wf)
    executor._outputs["n1"] = {"text": "你好"}
    inputs = executor._get_inputs("n2")
    assert inputs["text"] == "你好"


# --- prompt_template tests ---


@pytest.mark.asyncio
async def test_prompt_template_substitution():
    """text_input -> prompt_template -> output"""
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "Python"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "prompt_template", "data": {"template": "Write a {text} tutorial"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "text", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["text"] == "Write a Python tutorial"


# --- llm tests ---


@pytest.mark.asyncio
async def test_llm_node():
    """text_input -> llm -> output, mock the non-streaming httpx call."""
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "llm", "data": {"model": "test-model", "base_url": "http://localhost:8100", "stream": False}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "text", "target": "n3", "targetHandle": "text"},
        ],
    }

    # Mock httpx response: /v1/models probe (skipped when adapter is None falls through)
    # and POST /v1/chat/completions returning an OpenAI-compatible reply.
    post_resp = MagicMock()
    post_resp.status_code = 200
    post_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": "LLM reply"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    })

    get_resp = MagicMock()
    get_resp.status_code = 200
    get_resp.json = MagicMock(return_value={"data": [{"max_model_len": 4096}]})

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=post_resp)
    mock_client.get = AsyncMock(return_value=get_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.workflow_executor.httpx.AsyncClient", return_value=mock_client):
        executor = WorkflowExecutor(wf)
        result = await executor.execute()

    assert result["outputs"]["n2"]["text"] == "LLM reply"
    mock_client.post.assert_awaited_once()
    call_body = mock_client.post.await_args.kwargs["json"]
    assert call_body["model"] == "test-model"
    assert call_body["messages"][-1]["content"] == "hello"


# --- if_else tests ---


@pytest.mark.asyncio
async def test_if_else_true_branch():
    """Input contains condition string -> true branch gets text."""
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello world"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "if_else", "data": {"condition": "hello", "match_type": "contains"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "true", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["true"] == "hello world"
    assert result["outputs"]["n2"]["false"] == ""


@pytest.mark.asyncio
async def test_if_else_false_branch():
    """Input does NOT contain condition string -> false branch gets text."""
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "goodbye"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "if_else", "data": {"condition": "hello", "match_type": "contains"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "false", "target": "n3", "targetHandle": "text"},
        ],
    }
    executor = WorkflowExecutor(wf)
    result = await executor.execute()
    assert result["outputs"]["n2"]["true"] == ""
    assert result["outputs"]["n2"]["false"] == "goodbye"


# --- agent tests ---


@pytest.mark.asyncio
async def test_exec_agent_full():
    """Agent node loads config, assembles prompts, calls LLM."""
    mock_agent = {
        "name": "test-agent",
        "display_name": "Test",
        "model": {"engine_key": "test-model", "base_url": "http://localhost:8100"},
        "skills": [],
        "prompts": {
            "IDENTITY.md": "你是助手",
            "SOUL.md": "友好且专业",
            "AGENT.md": "回答用户问题",
        },
    }
    wf = {
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "你好"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "agent", "data": {"agent_name": "test-agent"}, "position": {"x": 200, "y": 0}},
            {"id": "n3", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
            {"id": "e2", "source": "n2", "sourceHandle": "text", "target": "n3", "targetHandle": "text"},
        ],
    }
    with patch("src.services.workflow_executor.agent_manager") as mock_am, \
         patch("src.services.llm_service.call_llm_with_tools", new_callable=AsyncMock) as mock_llm_tools, \
         patch("src.services.workflow_executor.call_llm", new_callable=AsyncMock):
        mock_am.get_agent.return_value = mock_agent
        # call_llm_with_tools returns a dict with "content" and optionally "tool_calls"
        mock_llm_tools.return_value = {"content": "你好！有什么可以帮你的？"}
        executor = WorkflowExecutor(wf)
        result = await executor.execute()

    assert result["outputs"]["n2"]["text"] == "你好！有什么可以帮你的？"
    # Verify system prompt was assembled from MD files
    call_args = mock_llm_tools.call_args
    system_in_messages = [m["content"] for m in call_args.kwargs.get("messages", []) if m["role"] == "system"]
    system_text = system_in_messages[0] if system_in_messages else ""
    assert "你是助手" in system_text
    assert "友好且专业" in system_text
