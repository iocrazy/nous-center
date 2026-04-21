"""Simple invokable nodes: text/multimodal in/out, passthrough."""

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
@register("resample")
@register("mixer")
@register("concat")
@register("bgm_mix")
class PassthroughNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return dict(inputs)


@register("multimodal_input")
class MultimodalInputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Multi-modal input — outputs text and optional images."""
        # Support both single image (legacy) and multiple images
        images = data.get("images") or []
        if not images:
            single = data.get("image", "")
            if single and single.startswith("data:"):
                images = [single]
        return {
            "text": data.get("text", ""),
            "image": images[0] if images else "",  # backward compat: first image
            "images": images,
            "audio": data.get("audio_data", ""),  # base64 data URL
        }


@register("output")
class OutputNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return inputs
