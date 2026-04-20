"""Simple invokable nodes: text in/out, passthrough."""

from src.services.nodes.registry import register


@register("text_input")
class TextInputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"text": data.get("text", "")}


@register("text_output")
class TextOutputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"text": inputs.get("text", "")}


@register("passthrough")
class PassthroughNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return dict(inputs)
