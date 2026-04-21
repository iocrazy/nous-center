import pytest

from src.services.nodes.base import InvokableNode
from src.services.nodes.text_io import TextInputNode, TextOutputNode, PassthroughNode


@pytest.mark.asyncio
async def test_text_input_node_returns_data_text():
    node = TextInputNode()
    assert isinstance(node, InvokableNode)
    result = await node.invoke({"text": "hello"}, {})
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_text_input_node_defaults_empty():
    node = TextInputNode()
    result = await node.invoke({}, {})
    assert result == {"text": ""}


@pytest.mark.asyncio
async def test_text_output_node_returns_inputs_text():
    node = TextOutputNode()
    result = await node.invoke({}, {"text": "out"})
    assert result == {"text": "out"}


@pytest.mark.asyncio
async def test_passthrough_node_returns_inputs():
    node = PassthroughNode()
    result = await node.invoke({}, {"key": "value"})
    assert result == {"key": "value"}
