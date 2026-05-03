"""Image nodes: image_generate (calls v2 IMAGE adapter) + image_output passthrough."""

from __future__ import annotations

import base64
import uuid

from src.services.inference.base import ImageRequest, LoRASpec
from src.services.nodes.registry import register


def _coerce_loras(raw) -> list[LoRASpec]:
    """data['loras'] arrives as either list[dict] (UI form) or list[LoRASpec]."""
    if not raw:
        return []
    out: list[LoRASpec] = []
    for entry in raw:
        if isinstance(entry, LoRASpec):
            out.append(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            out.append(LoRASpec(name=entry["name"], strength=float(entry.get("strength", 1.0))))
    return out


@register("image_generate")
class ImageGenerateNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        from src.services import workflow_executor as we

        prompt = inputs.get("prompt") or inputs.get("text") or data.get("prompt", "")
        if not prompt:
            raise we.ExecutionError("image_generate 节点缺少 prompt 输入")

        model_id = data.get("model_key") or data.get("model")
        if not model_id:
            raise we.ExecutionError("image_generate 节点缺少 model_key")

        if we._model_manager is None:
            raise we.ExecutionError("ModelManager 未初始化")

        adapter = await we._model_manager.get_loaded_adapter(model_id)

        req = ImageRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            negative_prompt=data.get("negative_prompt", ""),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 25)),
            seed=data.get("seed"),
            cfg_scale=float(data.get("cfg_scale", 7.0)),
            loras=_coerce_loras(data.get("loras")),
        )
        result = await adapter.infer(req)
        return {
            "image": base64.b64encode(result.data).decode(),
            "media_type": result.media_type,
            "width": result.metadata.get("width"),
            "height": result.metadata.get("height"),
            "steps": result.metadata.get("steps"),
            "seed": result.metadata.get("seed"),
            "loras": result.metadata.get("loras", []),
            "duration_ms": result.usage.latency_ms,
        }


@register("image_output")
class ImageOutputNode:
    """Render-only sink. Mirrors text_output / OutputNode shape so downstream
    workflow consumers (UI preview, signed-URL renderer in PR-6) see a
    consistent {image, media_type, width, height} envelope."""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {
            "image": inputs.get("image", ""),
            "media_type": inputs.get("media_type", "image/png"),
            "width": inputs.get("width"),
            "height": inputs.get("height"),
        }
