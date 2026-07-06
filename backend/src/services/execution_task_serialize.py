"""ExecutionTask → dict 序列化(任务面板/WS/API 共用)—— 下沉到 services 层
打破 services→api 反向依赖(workflow_runner 曾 import api.routes.execution_tasks._task_to_dict)。

纯逻辑:只依赖 ExecutionTask model + result envelope 形状。路由的 reaper
(collect_referenced_image_uuids,用 session)向下 import 这里的 _image_urls/_uuid。
"""
from __future__ import annotations

from src.models.execution_task import ExecutionTask

def _iter_node_outputs(result: object):
    """PR-5:真实 task.result 形态是 `{"outputs": {node_id: envelope, ...}}` —
    workflow_executor.execute() 包了一层 outputs(spec §3.3)。老的 detect_*_meta
    在 result.values() 找,只匹配「flat result.{node_id}.envelope」shape(不存在)。
    本 helper 同时兼容两种 shape:有 outputs 字段时 yield outputs.values();
    否则 yield result.values() 兜底(旧 fake / 测试用)。"""
    if not isinstance(result, dict):
        return
    outputs = result.get("outputs")
    if isinstance(outputs, dict):
        yield from outputs.values()
        return
    yield from result.values()


def _image_urls(result: object, limit: int = 8) -> list[str]:
    """抽出 result 里所有 image_url(签名 URL),供任务卡缩略图 + 历史画廊用
    (spec 2026-06-09 run-history PR-B)。最多 limit 张。"""
    urls: list[str] = []
    for v in _iter_node_outputs(result):
        if not isinstance(v, dict):
            continue
        u = v.get("image_url")
        if isinstance(u, str) and u:
            urls.append(u)
            if len(urls) >= limit:
                break
    return urls


def _uuid_from_image_url(url: str) -> str | None:
    """/files/images/{date}/{uuid}.{ext}?token&expires → {uuid}(文件 stem)。"""
    if not isinstance(url, str) or not url:
        return None
    name = url.split("?", 1)[0].rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    return stem or None


def _detect_image_meta(result: object) -> dict:
    """Pluck task_type + size from a workflow result by scanning for the
    image_output envelope shape. Stays None for non-image results so the
    UI can skip the badge entirely.
    """
    out: dict = {"task_type": None, "image_width": None, "image_height": None}
    for v in _iter_node_outputs(result):
        if not isinstance(v, dict):
            continue
        media_type = v.get("media_type")
        is_image = (
            (isinstance(media_type, str) and media_type.startswith("image/"))
            or "image_url" in v
        )
        if is_image:
            out["task_type"] = "image"
            w, h = v.get("width"), v.get("height")
            if isinstance(w, int) and isinstance(h, int):
                out["image_width"] = w
                out["image_height"] = h
            return out
    return out


def _detect_vision_meta(result: object) -> dict:
    """PR-1d:vision lane(LLM 节点带图/音输入)— LLMNode 返回 {text, usage, multimodal=True}。
    `multimodal=True` 是 PR-1d 在 LLMNode invoke/stream 加的标记字段;_detect_vision_meta
    据此把任务归 type=vision(spec ServiceType / 前端 紫橙渐变 + Vision 图标)。
    返 completion_tokens 供 callout 显示。"""
    out: dict = {"task_type": None, "vision_completion_tokens": None}
    for v in _iter_node_outputs(result):
        if not isinstance(v, dict):
            continue
        if v.get("multimodal") is True and isinstance(v.get("text"), str):
            out["task_type"] = "vision"
            usage = v.get("usage") or {}
            out["vision_completion_tokens"] = usage.get("completion_tokens")
            return out
    return out


def _detect_llm_meta(result: object) -> dict:
    """PR-1c:LLM workflow result envelope 识别 —— `text` 字符串 + `usage` dict 是
    LLMNode.invoke/stream 的标准返回形状(见 src/services/nodes/llm.py)。
    返 completion_tokens / prompt_tokens 供前端 callout 显示「47 tokens · 23 tok/s」。"""
    out: dict = {
        "task_type": None,
        "llm_prompt_tokens": None,
        "llm_completion_tokens": None,
    }
    for v in _iter_node_outputs(result):
        if not isinstance(v, dict):
            continue
        # LLM 节点返回 {text: str, usage: {...}, duration_ms: int}
        if isinstance(v.get("text"), str) and isinstance(v.get("usage"), dict):
            out["task_type"] = "llm"
            usage = v["usage"]
            out["llm_prompt_tokens"] = usage.get("prompt_tokens")
            out["llm_completion_tokens"] = usage.get("completion_tokens")
            return out
    return out


def _detect_tts_meta(result: object) -> dict:
    """PR-1b:TTS workflow result envelope 识别 —— audio/* media_type 或 audio_url。
    匹配 ImageBackend 落 image_output_storage 后的形状的 TTS 对应版本(audio_url + duration)。
    返 duration_seconds 供前端音频时长展示。"""
    out: dict = {"task_type": None, "audio_duration_seconds": None}
    for v in _iter_node_outputs(result):
        if not isinstance(v, dict):
            continue
        media_type = v.get("media_type")
        is_audio = (
            (isinstance(media_type, str) and media_type.startswith("audio/"))
            or "audio_url" in v
        )
        if is_audio:
            out["task_type"] = "tts"
            dur = v.get("duration_seconds") or (v.get("meta") or {}).get("duration_seconds")
            if isinstance(dur, (int, float)):
                out["audio_duration_seconds"] = float(dur)
            return out
    return out


def _task_to_dict(t: ExecutionTask) -> dict:
    d = {
        "id": str(t.id),
        "workflow_id": str(t.workflow_id) if t.workflow_id else None,
        "workflow_name": t.workflow_name,
        "status": t.status,
        "nodes_total": t.nodes_total,
        "nodes_done": t.nodes_done,
        "current_node": t.current_node,
        "result": t.result,
        "error": t.error,
        "duration_ms": t.duration_ms,
        # 历史参数 —— 前端「重跑(相同参数)」回填用(spec 2026-06-09 run-history PR-A)。
        "input_json": t.input_json,
        # 出图缩略(签名 URL 列表)—— 任务卡缩略图 + 历史画廊(PR-B)。
        "output_thumbnails": _image_urls(t.result),
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }
    # 顺序:image 先(workflow 多含 image_output)→ tts → vision(LLM 带图)→ llm(纯文本)。
    # 关键:vision 必须在 LLM 之前 —— 两者 envelope 都是 {text, usage},vision 多一个
    # multimodal=True 标记。先 vision 命中就归 vision,否则才 LLM。
    img_meta = _detect_image_meta(t.result)
    d.update(img_meta)
    if img_meta.get("task_type") is None:
        tts_meta = _detect_tts_meta(t.result)
        if tts_meta.get("task_type"):
            d["task_type"] = tts_meta["task_type"]
            d["audio_duration_seconds"] = tts_meta["audio_duration_seconds"]
        else:
            vision_meta = _detect_vision_meta(t.result)
            if vision_meta.get("task_type"):
                d["task_type"] = vision_meta["task_type"]
                d["vision_completion_tokens"] = vision_meta["vision_completion_tokens"]
            else:
                llm_meta = _detect_llm_meta(t.result)
                if llm_meta.get("task_type"):
                    d["task_type"] = llm_meta["task_type"]
                    d["llm_prompt_tokens"] = llm_meta["llm_prompt_tokens"]
                    d["llm_completion_tokens"] = llm_meta["llm_completion_tokens"]
    # PR-1a/1b/1c/1d(2026-05-27 任务面板重置 spec §State model):显式 `type` 字段
    # (image / tts / llm / vision),对应前端 ServiceType。type=None → 旧 fake / 未识别
    # workflow,前端 Other 兜底。
    d["type"] = d.get("task_type")
    return d
