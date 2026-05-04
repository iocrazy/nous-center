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
