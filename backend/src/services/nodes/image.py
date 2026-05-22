"""Image output node (render-only sink).

收敛后(spec 2026-05-21):全家桶 `image_generate` 已删除 —— 图像生成走细粒度图
(backend/nodes/flux2-components/:Load Diffusion/CLIP/VAE → Encode → KSampler →
VAE Decode),末端 `flux2_vae_decode` dispatch 到 image runner 经
`get_or_load_image_adapter` + `ImageSampler` 出图。`image_output` 作为终端展示节点
保留 —— VAE Decode 的 image 输出连到它。
"""

from __future__ import annotations

from src.services.nodes.registry import register


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
