"""ComfyUI history 产物分拣(仿 IC:扩展名定 kind,冗余 preview 图过滤)。"""
from __future__ import annotations

from dataclasses import dataclass

_KIND_BY_EXT = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "mp4": "video", "webm": "video", "mov": "video", "mkv": "video",
    "wav": "audio", "mp3": "audio", "flac": "audio", "ogg": "audio",
    "txt": "text", "json": "text", "srt": "text",
}
_PREVIEW_HINTS = ("previewimage", "comparer", "imagecompare")


@dataclass
class OutputItem:
    node_id: str
    class_type: str
    filename: str
    subfolder: str
    file_type: str
    kind: str


def classify_ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _KIND_BY_EXT.get(ext, "file")


def _is_preview(class_type: str) -> bool:
    ct = class_type.lower()
    return any(h in ct for h in _PREVIEW_HINTS)


def collect_outputs(history: dict, graph: dict) -> list[OutputItem]:
    cands: list[OutputItem] = []
    for node_id, node_out in (history.get("outputs") or {}).items():
        ct = str((graph.get(str(node_id)) or {}).get("class_type") or "")
        for key in ("images", "videos", "audio", "gifs", "files"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                cands.append(OutputItem(
                    node_id=str(node_id), class_type=ct,
                    filename=str(item["filename"]),
                    subfolder=str(item.get("subfolder", "")),
                    file_type=str(item.get("type", "output")),
                    kind=classify_ext(str(item["filename"]))))
    has_primary_image = any(c.kind == "image" and not _is_preview(c.class_type) for c in cands)
    return [c for c in cands
            if not (c.kind == "image" and has_primary_image and _is_preview(c.class_type))]
