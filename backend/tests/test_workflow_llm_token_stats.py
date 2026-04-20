"""CRITICAL REGRESSION: existing frontend token stats must keep working after node migration.

Wave 1 Task 4 Subtask 4.3: LLMNode must implement both Invokable (non-stream)
and Streamable (stream) protocols. The stream path must call on_token per chunk
(no more global _on_progress_ref) and must return a final dict with usage stats.
"""

import pytest

from src.services.nodes.base import InvokableNode, StreamableNode
from src.services.nodes.llm import LLMNode


def test_llm_node_implements_both_protocols():
    node = LLMNode()
    assert isinstance(node, InvokableNode)
    assert isinstance(node, StreamableNode)


@pytest.mark.asyncio
async def test_llm_node_stream_invokes_on_token_per_chunk(mock_llm_stream_v2):
    """on_token should be awaited once per chunk, not via global ref."""
    captured: list[str] = []

    async def on_token(t: str):
        captured.append(t)

    node = LLMNode()
    result = await node.stream(
        data={"_node_id": "llm-1", "model": "qwen3.5"},
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        on_token=on_token,
    )
    assert captured == ["hel", "lo"]  # from fake_stream fixture
    assert result["usage"]["total_tokens"] == 4
    assert "text" in result  # 最终拼好的 assistant text


@pytest.mark.asyncio
async def test_llm_node_invoke_non_stream(mock_llm_nonstream):
    """Non-streaming invoke path."""
    node = LLMNode()
    result = await node.invoke(
        data={"_node_id": "llm-2", "model": "qwen3.5", "stream": False},
        inputs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert result["text"] == "non-stream response"
    assert result["usage"]["total_tokens"] > 0
