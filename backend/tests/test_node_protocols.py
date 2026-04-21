from src.services.nodes.base import (
    InvokableNode,
    StreamableNode,
)


def test_protocols_are_runtime_checkable():
    # 验证 Protocol 声明了 @runtime_checkable
    assert hasattr(InvokableNode, "_is_runtime_protocol")


def test_invokable_protocol_shape():
    class _Good:
        async def invoke(self, data, inputs):
            return {}
    assert isinstance(_Good(), InvokableNode)


def test_invokable_protocol_rejects_missing_method():
    class _Bad:
        pass
    assert not isinstance(_Bad(), InvokableNode)


def test_streamable_protocol_shape():
    class _Stream:
        async def stream(self, data, inputs, on_token):
            return {}
    assert isinstance(_Stream(), StreamableNode)
