import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio

async def test_list_tasks_empty(db_client: AsyncClient):
    resp = await db_client.get("/api/v1/tasks")
    assert resp.status_code == 200
    assert resp.json() == []

async def test_record_task(db_client: AsyncClient):
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "test_workflow",
        "status": "completed",
        "nodes_total": 3,
        "nodes_done": 3,
        "duration_ms": 1234,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow_name"] == "test_workflow"
    assert data["status"] == "completed"
    assert data["duration_ms"] == 1234

async def test_record_task_invalid_status(db_client: AsyncClient):
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "test",
        "status": "INVALID",
    })
    assert resp.status_code == 400

async def test_get_task(db_client: AsyncClient):
    # Create a task first
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "get_test",
        "status": "completed",
        "nodes_total": 1,
        "nodes_done": 1,
    })
    task_id = resp.json()["id"]

    # Get it
    resp = await db_client.get(f"/api/v1/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["workflow_name"] == "get_test"

async def test_get_task_not_found(db_client: AsyncClient):
    resp = await db_client.get("/api/v1/tasks/999999999")
    assert resp.status_code == 404

async def test_delete_task(db_client: AsyncClient):
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "delete_test",
        "status": "failed",
    })
    task_id = resp.json()["id"]

    resp = await db_client.delete(f"/api/v1/tasks/{task_id}")
    assert resp.status_code == 200

    resp = await db_client.get(f"/api/v1/tasks/{task_id}")
    assert resp.status_code == 404

async def test_cancel_task(db_client: AsyncClient):
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "cancel_test",
        "status": "running",
    })
    task_id = resp.json()["id"]

    resp = await db_client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert resp.status_code == 200

    resp = await db_client.get(f"/api/v1/tasks/{task_id}")
    assert resp.json()["status"] == "cancelled"

async def test_cancel_completed_task_fails(db_client: AsyncClient):
    resp = await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "cancel_fail",
        "status": "completed",
    })
    task_id = resp.json()["id"]

    resp = await db_client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert resp.status_code == 400

async def test_list_tasks_with_filter(db_client: AsyncClient):
    await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "filter1", "status": "completed",
    })
    await db_client.post("/api/v1/tasks/record", json={
        "workflow_name": "filter2", "status": "failed",
    })

    resp = await db_client.get("/api/v1/tasks?status=failed")
    tasks = resp.json()
    assert all(t["status"] == "failed" for t in tasks)


def test_detect_image_meta_recognizes_image_output_envelope():
    from src.api.routes.execution_tasks import _detect_image_meta

    result = {
        "node-abc": {"text": "hi"},
        "node-img": {
            "image_url": "/v1/images/x.png?sig=...",
            "media_type": "image/png",
            "width": 1024,
            "height": 1024,
        },
    }
    meta = _detect_image_meta(result)
    assert meta == {"task_type": "image", "image_width": 1024, "image_height": 1024}


def test_detect_image_meta_returns_none_for_text_only_result():
    from src.api.routes.execution_tasks import _detect_image_meta

    assert _detect_image_meta({"a": {"text": "hello"}}) == {
        "task_type": None,
        "image_width": None,
        "image_height": None,
    }
    assert _detect_image_meta(None) == {
        "task_type": None,
        "image_width": None,
        "image_height": None,
    }


def test_detect_image_meta_handles_image_without_dimensions():
    from src.api.routes.execution_tasks import _detect_image_meta

    meta = _detect_image_meta({"out": {"media_type": "image/jpeg"}})
    assert meta["task_type"] == "image"
    assert meta["image_width"] is None
    assert meta["image_height"] is None


def test_detect_vision_meta_recognizes_multimodal_marker():
    """PR-1d:LLMNode 标了 multimodal=True → _detect_vision_meta 归 type=vision。"""
    from src.api.routes.execution_tasks import _detect_vision_meta

    result = {
        "vis-1": {
            "text": "a cat",
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
            "multimodal": True,
        }
    }
    meta = _detect_vision_meta(result)
    assert meta == {"task_type": "vision", "vision_completion_tokens": 8}


def test_detect_vision_meta_returns_none_for_pure_llm():
    """multimodal=False 或缺失 → 不算 vision(由 _detect_llm_meta 兜底)。"""
    from src.api.routes.execution_tasks import _detect_vision_meta

    assert _detect_vision_meta({"a": {"text": "x", "usage": {}}})["task_type"] is None
    assert _detect_vision_meta(
        {"a": {"text": "x", "usage": {}, "multimodal": False}})["task_type"] is None


def test_task_to_dict_vision_takes_precedence_over_llm():
    """PR-1d 关键:vision 检测顺序在 LLM 之前 —— 同样 {text, usage} envelope,有
    multimodal=True 就归 vision,无就归 llm。"""
    from datetime import datetime, timezone

    from src.api.routes.execution_tasks import _task_to_dict
    from src.models.execution_task import ExecutionTask

    now = datetime.now(timezone.utc)
    t_vision = ExecutionTask(
        id=6, workflow_id=1, workflow_name="vqa", status="completed",
        nodes_total=2, nodes_done=2, current_node=None,
        result={"vis-1": {
            "text": "a cat",
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
            "multimodal": True,
        }},
        error=None, duration_ms=900, created_at=now, updated_at=now,
    )
    d = _task_to_dict(t_vision)
    assert d["type"] == "vision"
    assert d["vision_completion_tokens"] == 8
    # 不应该被误标为 llm
    assert d.get("llm_completion_tokens") is None


def test_detect_llm_meta_recognizes_text_and_usage_envelope():
    """PR-1c:LLM workflow result envelope 识别(text + usage)。"""
    from src.api.routes.execution_tasks import _detect_llm_meta

    result = {
        "llm-node": {
            "text": "answer",
            "usage": {"prompt_tokens": 12, "completion_tokens": 47, "total_tokens": 59},
            "duration_ms": 1234,
        }
    }
    meta = _detect_llm_meta(result)
    assert meta == {
        "task_type": "llm",
        "llm_prompt_tokens": 12,
        "llm_completion_tokens": 47,
    }


def test_detect_llm_meta_returns_none_for_non_llm():
    """text 但无 usage / usage 但无 text → 不算 LLM。"""
    from src.api.routes.execution_tasks import _detect_llm_meta

    # 只有 text(text_input 节点)
    assert _detect_llm_meta({"a": {"text": "hi"}})["task_type"] is None
    # 只有 usage(理论上不会发生但保护)
    assert _detect_llm_meta({"a": {"usage": {}}})["task_type"] is None


def test_task_to_dict_exposes_type_field_for_llm_result():
    """PR-1c:LLM-only result → type=\"llm\" + llm_prompt/completion_tokens。"""
    from datetime import datetime, timezone

    from src.api.routes.execution_tasks import _task_to_dict
    from src.models.execution_task import ExecutionTask

    now = datetime.now(timezone.utc)
    t_llm = ExecutionTask(
        id=5, workflow_id=1, workflow_name="chat", status="completed",
        nodes_total=2, nodes_done=2, current_node=None,
        result={"llm-1": {
            "text": "hi",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "duration_ms": 500,
        }},
        error=None, duration_ms=500, created_at=now, updated_at=now,
    )
    d = _task_to_dict(t_llm)
    assert d["type"] == "llm"
    assert d["task_type"] == "llm"
    assert d["llm_prompt_tokens"] == 5
    assert d["llm_completion_tokens"] == 10


def test_detect_tts_meta_recognizes_audio_envelope():
    """PR-1b:TTS workflow result envelope 识别(audio_url + duration_seconds)。"""
    from src.api.routes.execution_tasks import _detect_tts_meta

    result = {
        "node-tts": {
            "audio_url": "/v1/audio/x.wav?sig=...",
            "media_type": "audio/wav",
            "duration_seconds": 3.42,
        }
    }
    meta = _detect_tts_meta(result)
    assert meta == {"task_type": "tts", "audio_duration_seconds": 3.42}


def test_detect_tts_meta_via_meta_envelope():
    """PR-1b:duration 可能在 meta 子对象里(runner_process outputs envelope 形状)。"""
    from src.api.routes.execution_tasks import _detect_tts_meta

    result = {
        "node-tts": {
            "media_type": "audio/wav",
            "meta": {"duration_seconds": 5.0, "sample_rate": 24000},
        }
    }
    meta = _detect_tts_meta(result)
    assert meta["task_type"] == "tts"
    assert meta["audio_duration_seconds"] == 5.0


def test_detect_tts_meta_returns_none_for_image_result():
    """PR-1b:image result(audio media_type 缺)→ TTS 检测 None。"""
    from src.api.routes.execution_tasks import _detect_tts_meta

    assert _detect_tts_meta({"x": {"media_type": "image/png"}})["task_type"] is None


def test_task_to_dict_exposes_type_field_for_tts_result():
    """PR-1b:TTS-only result → _task_to_dict 加 type=\"tts\" + audio_duration_seconds。"""
    from datetime import datetime, timezone

    from src.api.routes.execution_tasks import _task_to_dict
    from src.models.execution_task import ExecutionTask

    now = datetime.now(timezone.utc)
    t_tts = ExecutionTask(
        id=4, workflow_id=1, workflow_name="w", status="completed",
        nodes_total=1, nodes_done=1, current_node=None,
        result={"node-tts": {
            "audio_url": "/v1/audio/x.wav",
            "media_type": "audio/wav",
            "duration_seconds": 2.5,
        }},
        error=None, duration_ms=200, created_at=now, updated_at=now,
    )
    d = _task_to_dict(t_tts)
    assert d["type"] == "tts"
    assert d["task_type"] == "tts"
    assert d["audio_duration_seconds"] == 2.5


def test_task_to_dict_exposes_type_field_for_image_result():
    """PR-1a:_task_to_dict 加显式 `type` 字段(对齐 spec State model ServiceType)。
    image result → type="image";text-only / None result → type=None(前端 Other 兜底)。
    沿用 _detect_image_meta 的检测,tts/llm/vision 由后续 PR-1b/c/d 扩展。"""
    from datetime import datetime, timezone

    from src.api.routes.execution_tasks import _task_to_dict
    from src.models.execution_task import ExecutionTask

    now = datetime.now(timezone.utc)
    # image result
    t_img = ExecutionTask(
        id=1, workflow_id=1, workflow_name="w", status="completed",
        nodes_total=1, nodes_done=1, current_node=None,
        result={"out": {"image_url": "/x.png", "media_type": "image/png",
                        "width": 1024, "height": 1024}},
        error=None, duration_ms=100, created_at=now, updated_at=now,
    )
    d = _task_to_dict(t_img)
    assert d["type"] == "image"
    assert d["task_type"] == "image"  # 旧字段保留(向后兼容)

    # text-only result → type=None
    t_text = ExecutionTask(
        id=2, workflow_id=1, workflow_name="w2", status="completed",
        nodes_total=1, nodes_done=1, current_node=None,
        result={"out": {"text": "hi"}},
        error=None, duration_ms=50, created_at=now, updated_at=now,
    )
    assert _task_to_dict(t_text)["type"] is None

    # None result → type=None
    t_none = ExecutionTask(
        id=3, workflow_id=1, workflow_name="w3", status="queued",
        nodes_total=0, nodes_done=0, current_node=None,
        result=None, error=None, duration_ms=None,
        created_at=now, updated_at=now,
    )
    assert _task_to_dict(t_none)["type"] is None
