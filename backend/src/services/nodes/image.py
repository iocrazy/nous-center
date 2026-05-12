"""Image nodes: image_generate (composes Lane C helpers) + image_output passthrough.

ADMIN_SESSION_SECRET MUST be set when image workflows run — the signed
URL is the canonical and only render path. write_image returns url=None
when the secret is missing, which surfaces here as MissingSecretError.

V1' Lane D P5: image_generate is now a thin orchestrator over the same
encode_prompt / sample / vae_decode helpers that back the component-node
set under backend/nodes/flux2-components/. Integrated and component
paths share one code path, so a bug fix or perf tweak applied to either
flows through to both. The visible output schema is preserved byte-for-
byte against the old adapter.infer route so existing workflows + the
exposed_outputs allowlist in workflow_publish.py keep working.
"""

from __future__ import annotations

import asyncio
import io
import secrets
import time

from src.services.image_output_storage import write_image
from src.services.inference.base import LoRASpec
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
        from src.services.inference.image_diffusers import (
            encode_prompt, sample, vae_decode,
        )

        prompt = inputs.get("prompt") or inputs.get("text") or data.get("prompt", "")
        if not prompt:
            raise we.ExecutionError("image_generate 节点缺少 prompt 输入")

        model_id = data.get("model_key") or data.get("model")
        if not model_id:
            raise we.ExecutionError("image_generate 节点缺少 model_key")

        if we._model_manager is None:
            raise we.ExecutionError("ModelManager 未初始化")

        adapter = await we._model_manager.get_loaded_adapter(model_id)

        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = int(data.get("steps", 25))
        cfg_scale = float(data.get("cfg_scale", 7.0))
        loras = _coerce_loras(data.get("loras"))

        # ComfyUI-style: draw a fresh 64-bit seed when none supplied so every
        # run is reproducible (the seed is echoed back in metadata). Use
        # secrets.randbelow for the same crypto-rand semantics adapter.infer
        # had before P5.
        seed_raw = data.get("seed")
        seed = int(seed_raw) if seed_raw not in (None, "") else secrets.randbelow(2**63)

        # set_active_loras runs on the loop because the apply step itself is
        # fast (<100ms) and the offload-disable+enable cycle benefits from
        # staying observable in one coroutine context — matches what the
        # Lane C KSampler node does.
        adapter.set_active_loras(loras)

        import torch
        generator = torch.Generator(device=adapter.device).manual_seed(seed)

        t0 = time.monotonic()
        cond = await asyncio.to_thread(encode_prompt, adapter.pipe, prompt)
        latents = await asyncio.to_thread(
            sample, adapter.pipe, cond,
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )
        image = await asyncio.to_thread(vae_decode, adapter.pipe, latents)
        latency_ms = int((time.monotonic() - t0) * 1000)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        ttl = int(data.get("url_ttl_seconds", 3600))
        record = write_image(buf.getvalue(), ext="png", ttl_seconds=ttl)
        if record["url"] is None:
            raise we.ExecutionError(
                "image_generate 需要 ADMIN_SESSION_SECRET 才能签名输出 URL — "
                "请在 backend/.env 配置后重启 backend"
            )
        return {
            "media_type": "image/png",
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "loras": [{"name": s.name, "strength": s.strength} for s in loras],
            "duration_ms": latency_ms,
            "image_url": record["url"],
            "image_uuid": record["uuid"],
            "image_expires": record["expires"],
        }


@register("image_output")
class ImageOutputNode:
    """Render-only sink. Stable envelope: {image_url, media_type, width, height}.
    image_url is the canonical (and only) render path — the signed URL HMAC'd
    against ADMIN_SESSION_SECRET.
    """

    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {
            "image_url": inputs.get("image_url"),
            "media_type": inputs.get("media_type", "image/png"),
            "width": inputs.get("width"),
            "height": inputs.get("height"),
        }
