"""ComfyUI workflow bridge nodes (Task 6).

`comfyui_workflow` is the single inline node every 模板即服务 bridge snapshot
runs through (see `src.api.routes.comfy_templates._bridge_snapshot`): it loads
the raw ComfyUI API-format graph + the service's `exposed_inputs` mapping,
patches user-supplied values into the graph (uploading `media`-typed data
URIs, rolling `random`-marked seeds), submits + waits on the ComfyUI sidecar
serialized through a module-level semaphore (sidecar 显存独占 — one render at
a time), then persists every collected output via `write_media` and returns a
stable envelope. `video_output` is the terminal sink node mirroring
`image.py`'s `image_output` (原样透传).
"""
from __future__ import annotations

import asyncio
import base64
import os
import secrets
import tempfile
from copy import deepcopy
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import undefer

from src.models.comfy_template import ComfyTemplate
from src.models.database import get_session_factory
from src.models.service_instance import ServiceInstance
from src.services.comfy.client import ComfyClient
from src.services.comfy.outputs import collect_outputs
from src.services.comfy.thumbnail import extract_first_frame
from src.services.image_output_storage import write_image as _write_media_sync
from src.services.nodes.registry import register

# Sidecar 显存独占:一次只能有一个渲染在跑(submit+wait 串行化,download/storage 不占)。
_SEM = asyncio.Semaphore(1)


def get_client() -> ComfyClient:
    """Factory function to create a ComfyClient instance.
    Tests monkeypatch this to inject a mock client (同 route 层 get_client 惯例)。"""
    return ComfyClient()


async def write_media(data: bytes, *, ext: str, ttl_seconds: int = 86400) -> dict:
    """Async wrapper around the sync `image_output_storage.write_image` — disk
    I/O runs off the event loop via a thread so the render-serial semaphore
    above isn't held while we write files."""
    return await asyncio.to_thread(_write_media_sync, data, ext=ext, ttl_seconds=ttl_seconds)


async def load_template(template_id) -> tuple[dict, list[dict]]:
    """Load `(workflow_json, exposed_inputs)` for a comfy_templates row.

    exposed_inputs live on the paired ServiceInstance (source_type=
    "comfy_template", source_id=template_id) — same linkage the route layer
    uses in `comfy_templates._get_template_and_service`.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        tpl = await session.get(ComfyTemplate, int(template_id))
        if tpl is None:
            raise ValueError(f"模板不存在:{template_id}")
        stmt = (
            select(ServiceInstance)
            .options(undefer(ServiceInstance.exposed_inputs))
            .where(ServiceInstance.source_type == "comfy_template")
            .where(ServiceInstance.source_id == tpl.id)
        )
        svc = (await session.execute(stmt)).scalar_one_or_none()
        exposed = list(svc.exposed_inputs or []) if svc is not None else []
        return dict(tpl.workflow_json or {}), exposed


def _decode_data_uri(value: str) -> tuple[bytes, str, str]:
    """`data:image/png;base64,...` → (raw bytes, ext, mime)。"""
    header, _, b64data = value.partition(",")
    mime = header[len("data:"):].split(";")[0] or "image/png"
    ext = mime.rsplit("/", 1)[-1] or "png"
    return base64.b64decode(b64data), ext, mime


@register("comfyui_workflow")
class ComfyUIWorkflowNode:
    """Bridge node: patch exposed params into the ComfyUI graph, run it on
    the sidecar, persist outputs. See module docstring for the full flow."""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        template_id = data.get("template_id")
        graph_template, exposed_params = await load_template(template_id)
        graph = deepcopy(graph_template)
        client = get_client()
        seed = None

        for m in exposed_params:
            key = m["key"]
            value = data.get(key, m.get("default"))
            node_id = str(m.get("comfy_node_id"))
            input_name = m.get("comfy_input")

            if m.get("type") == "media" and isinstance(value, str) and value.startswith("data:"):
                raw, ext, mime = _decode_data_uri(value)
                filename = f"{secrets.token_hex(8)}.{ext}"
                value = await client.upload_image(filename, raw, mime)

            if m.get("random") and (value is None or value == ""):
                value = secrets.randbelow(2 ** 32)
                seed = value

            if m.get("required") and (value is None or value == ""):
                raise ValueError(f"缺少必填参数:{key}")

            if value is None:
                continue

            node = graph.get(node_id)
            if node is not None:
                node.setdefault("inputs", {})[input_name] = value

        async with _SEM:
            prompt_id = await client.submit(graph)
            timeout_s = float(os.getenv("NOUS_COMFY_TIMEOUT", "14400"))
            history = await client.wait(prompt_id, timeout_s=timeout_s)

        outputs = collect_outputs(history, graph)
        items: list[dict] = []
        video_url: str | None = None
        thumbnails: list[str] = []

        for o in outputs:
            raw_bytes = await client.download(
                {"filename": o.filename, "subfolder": o.subfolder, "type": o.file_type})
            ext = o.filename.rsplit(".", 1)[-1].lower() if "." in o.filename else "bin"
            stored = await write_media(raw_bytes, ext=ext)
            items.append({
                "url": stored["url"], "kind": o.kind, "filename": o.filename,
                "node_id": o.node_id, "class_type": o.class_type,
            })

            if o.kind == "video":
                if video_url is None:
                    video_url = stored["url"]
                with tempfile.TemporaryDirectory(prefix="nous-bridge-vid-") as tmp_dir:
                    vid_path = Path(tmp_dir) / f"src.{ext}"
                    vid_path.write_bytes(raw_bytes)
                    frame = await extract_first_frame(vid_path)
                if frame is not None:
                    thumb = await write_media(frame, ext="png")
                    thumbnails.append(thumb["url"])

        return {"items": items, "video_url": video_url, "thumbnails": thumbnails, "seed": seed}


@register("video_output")
class VideoOutputNode:
    """Terminal sink — 原样透传上游产物(image_output 的 video 对应版)。"""

    async def invoke(self, data: dict, inputs: dict) -> dict:
        return inputs.get("outputs") or inputs
