"""image-color-match 节点 executor — 色彩匹配(目标图色彩向参考图迁移)。

image_color_match 是 **inline 节点**(主进程 event loop,CPU:color-matcher transfer + 线性混合 + 写盘):
两路上游 image_url(签名 URL)→ 解析回本地磁盘读图 → ColorMatcher().transfer(src=target, ref=ref,
method) → strength 线性插值 → 落盘签 URL(同 flux2_vae_decode 出图路径)。

对齐 ComfyUI-KJNodes ColorMatch(image_nodes.py):mkl 等方法 + strength 混合
`result = target + strength*(matched - target)`。纯图像处理无模型,故 inline 不吃 GPU。
"""
from __future__ import annotations


def _resolve_local_path(image_url: str) -> str:
    """上游 image_url(签名 URL /files/images/<date>/<uuid>.<ext>?token=...)→ 本地磁盘路径。

    backend 与 outputs 同机共享 NAS_OUTPUTS_PATH —— 按 date/uuid/ext 解析磁盘文件直接读,
    免 HTTP 回环 + token 验证(图本就是本工作流上游刚生成的)。非 /files/ 形态(本地路径)原样返回。
    与 runner._resolve_input_image_path 同语义(inline 主进程版,各自含,blast radius 隔离)。
    """
    s = str(image_url)
    if not (s.startswith("/files/") or "/files/images/" in s):
        return s
    from urllib.parse import urlparse  # noqa: PLC0415

    from src.services.image_output_storage import resolve_path  # noqa: PLC0415
    path = urlparse(s).path
    parts = path.strip("/").split("/")
    date = parts[-2]
    uuid_str, _, ext = parts[-1].rpartition(".")
    return str(resolve_path(date, uuid_str, ext or "png"))


async def exec_color_match(data: dict, inputs: dict) -> dict:
    """两路 image → color-matcher transfer + strength 混合 → 落盘签 URL。"""
    import io  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from color_matcher import ColorMatcher  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    target_url = inputs.get("image_target") or inputs.get("image_url")
    ref_url = inputs.get("image_ref")
    if not target_url:
        raise RuntimeError("图像色彩匹配:缺目标图输入(image_target)")
    if not ref_url:
        raise RuntimeError("图像色彩匹配:缺参考图输入(image_ref)")

    tgt = Image.open(_resolve_local_path(target_url)).convert("RGB")
    ref = Image.open(_resolve_local_path(ref_url)).convert("RGB")
    tgt_np = np.asarray(tgt, dtype=np.float32) / 255.0
    ref_np = np.asarray(ref, dtype=np.float32) / 255.0

    # 引擎边界归一(2026-06-11 体检):method 白名单(与 node.yaml options 同步,非法值
    # ColorMatcher 深处崩报错难读)+ strength clamp。合法值原样(零回归)。
    _methods = {"mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"}
    method = str(data.get("method") or "mkl")
    if method not in _methods:
        method = "mkl"
    matched = ColorMatcher().transfer(src=tgt_np, ref=ref_np, method=method)

    # strength 线性插值(对齐 KJNodes:result = src + strength*(matched - src))
    strength = max(0.0, min(1.0, float(data.get("strength", 0.65))))
    result = tgt_np + strength * (matched - tgt_np)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    raw = buf.getvalue()

    from src.config import get_settings  # noqa: PLC0415
    from src.services.image_output_storage import write_image  # noqa: PLC0415
    ttl = int(get_settings().IMAGE_URL_TTL_SECONDS)
    record = write_image(raw, ext="png", ttl_seconds=ttl)

    h, w = result.shape[:2]
    return {
        "image_url": record["url"],
        "image_uuid": record["uuid"],
        "image_expires": record["expires"],
        "media_type": "image/png",
        "width": int(w),
        "height": int(h),
    }


EXECUTORS = {
    "image_color_match": exec_color_match,
}
