"""image-io 节点 executor — 图像输入(上传图 → 落盘签 URL → image 端口)。

image_input 是 **inline 节点**(主进程 event loop,CPU:base64 解码 + 写盘 + 量宽高):
把前端上传的 base64 data URI 落盘到 image_output_storage(同 flux2_vae_decode 出图路径),
产出 `{image_url, media_type, width, height}` —— 下游 image→image 节点(SeedVR2 超分)
经 inputs 拿到 image_url(签名 URL),runner 端 _resolve_input_image_path 解析回本地磁盘读图。

为何落盘签 URL 而非直接透传 base64:与 flux2_vae_decode 一致 —— base64 大图走 runner
msgpack pipe 是反模式(可能几 MB)。落盘后只过一个签名 URL 字符串。
"""
from __future__ import annotations

import base64


async def exec_image_input(data: dict, inputs: dict) -> dict:
    """上传图(data.image:base64 data URI)→ 落盘签 URL → {image_url, media_type, width, height}。"""
    src = data.get("image") or ""
    if not src or not isinstance(src, str) or not src.startswith("data:"):
        raise RuntimeError("图像输入:未上传图(data.image 应为 base64 data URI 'data:image/...;base64,...')")

    # "data:image/png;base64,...." → media_type + 原始 bytes。
    header, _, b64 = src.partition(",")
    media_type = "image/png"
    if header.startswith("data:") and ";" in header:
        media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
    raw = base64.b64decode(b64)

    # 落盘签 URL(NAS_OUTPUTS_PATH + HMAC),同 flux2_vae_decode 出图。
    from src.services.image_output_storage import write_image  # noqa: PLC0415
    ext = media_type.split("/", 1)[1].split("+", 1)[0] or "png"
    ttl = int(data.get("url_ttl_seconds") or 3600)
    record = write_image(raw, ext=ext, ttl_seconds=ttl)

    # 量宽高(可选,失败不致命 —— 下游超分不依赖,只为 UI/meta)。
    width = height = None
    try:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 — best-effort metadata
        pass

    return {
        "image_url": record["url"],
        "image_uuid": record["uuid"],
        "image_expires": record["expires"],
        "media_type": media_type,
        "width": width,
        "height": height,
    }


EXECUTORS = {
    "image_input": exec_image_input,
}
