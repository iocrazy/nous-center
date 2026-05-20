"""PR-4 image component loader nodes — pure descriptor producers (inline).

These run in the backend event loop (node_routing default = inline, no GPU).
Each emits a plain descriptor dict; no tensors cross the wire. The runner
subprocess later materializes descriptors into ComponentSpec + ImageSampler
(spec §3.2). Output port names (unet/clip/vae) match image_generate's input
ports so WorkflowExecutor._get_inputs lands them in merged_inputs under those
keys (spec §5.4). Frontend palette + forms are PR-5.
"""
from __future__ import annotations

from src.services.nodes.registry import register

_AUTO = "auto"


@register("image_unet_load")
class ImageUnetLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"unet": {
            "kind": "unet",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
            "adapter_arch": data.get("adapter_arch") or "flux2",
            "loras": [],
        }}


@register("image_clip_load")
class ImageClipLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"clip": {
            "kind": "clip",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
            "clip_arch": data.get("clip_arch") or "flux2",
        }}


@register("image_vae_load")
class ImageVaeLoadNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {"vae": {
            "kind": "vae",
            "file": data["file"],
            "device": data.get("device") or _AUTO,
            "dtype": data.get("dtype") or "bfloat16",
        }}


@register("image_lora_apply")
class ImageLoraApplyNode:
    """Chainable: input unet descriptor → output unet descriptor with one more
    LoRA appended. bypass=True passes the upstream descriptor straight through
    (spec §4.4)."""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        from src.services.workflow_executor import ExecutionError

        upstream = inputs.get("unet")
        if not isinstance(upstream, dict) or upstream.get("kind") != "unet":
            raise ExecutionError("image_lora_apply 需要上游 unet 描述符输入(连 image_unet_load 或上一个 image_lora_apply)")
        if data.get("bypass"):
            return {"unet": upstream}
        import os
        lora_path = data.get("lora_path")
        lora_name = data.get("lora_file") or (os.path.basename(lora_path) if lora_path else None)
        if not lora_name:
            from src.services.workflow_executor import ExecutionError
            raise ExecutionError("image_lora_apply 需要 lora_path 或 lora_file")
        appended = {
            "name": lora_name,
            "path": lora_path,
            "strength": float(data.get("strength", 1.0)),
        }
        return {"unet": {**upstream, "loras": [*upstream.get("loras", []), appended]}}
