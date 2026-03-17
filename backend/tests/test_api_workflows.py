import pytest


pytestmark = pytest.mark.anyio


async def test_create_workflow(db_client):
    resp = await db_client.post("/api/v1/workflows", json={
        "name": "测试流程",
        "nodes": [{"id": "n1", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}}],
        "edges": [],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "测试流程"
    assert data["status"] == "draft"
    assert len(data["nodes"]) == 1
    assert "id" in data


async def test_list_workflows(db_client):
    await db_client.post("/api/v1/workflows", json={"name": "w1"})
    await db_client.post("/api/v1/workflows", json={"name": "w2", "is_template": True})
    resp = await db_client.get("/api/v1/workflows")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_list_workflows_filter_template(db_client):
    await db_client.post("/api/v1/workflows", json={"name": "w1"})
    await db_client.post("/api/v1/workflows", json={"name": "w2", "is_template": True})
    resp = await db_client.get("/api/v1/workflows?is_template=true")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["is_template"] is True


async def test_get_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "w1"


async def test_get_workflow_not_found(db_client):
    resp = await db_client.get("/api/v1/workflows/999999")
    assert resp.status_code == 404


async def test_update_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "old"})
    wf_id = create.json()["id"]
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"name": "new"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new"


async def test_update_workflow_nodes(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1", "nodes": []})
    wf_id = create.json()["id"]
    new_nodes = [{"id": "n1", "type": "text_input", "data": {}, "position": {"x": 0, "y": 0}}]
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"nodes": new_nodes})
    assert resp.status_code == 200
    assert len(resp.json()["nodes"]) == 1


async def test_delete_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    resp = await db_client.delete(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 204
    resp2 = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp2.status_code == 404


async def test_publish_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={
        "name": "发布测试",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [{"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "audio"}],
    })
    wf_id = create.json()["id"]
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    assert resp.status_code == 200
    data = resp.json()
    assert data["instance_id"] is not None
    assert data["endpoint"] is not None

    wf = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert wf.json()["status"] == "published"


async def test_publish_not_found(db_client):
    resp = await db_client.post("/api/v1/workflows/999999/publish")
    assert resp.status_code == 404


async def test_unpublish_workflow(db_client):
    create = await db_client.post("/api/v1/workflows", json={"name": "w1"})
    wf_id = create.json()["id"]
    await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/unpublish")
    assert resp.status_code == 200

    wf = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert wf.json()["status"] == "draft"


async def test_run_published_workflow(db_client):
    """POST /v1/instances/{id}/run executes a published workflow."""
    create = await db_client.post("/api/v1/workflows", json={
        "name": "run-test",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
        ],
    })
    wf_id = create.json()["id"]
    pub = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    instance_id = pub.json()["instance_id"]

    key_resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={"label": "test"})
    api_key = key_resp.json()["key"]

    resp = await db_client.post(
        f"/v1/instances/{instance_id}/run",
        json={},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "outputs" in data


async def test_workflow_full_lifecycle(db_client):
    """Create -> Update -> Publish -> Run -> Unpublish -> Delete"""
    # Create
    resp = await db_client.post("/api/v1/workflows", json={
        "name": "lifecycle",
        "nodes": [
            {"id": "n1", "type": "text_input", "data": {"text": "test"}, "position": {"x": 0, "y": 0}},
            {"id": "n2", "type": "output", "data": {}, "position": {"x": 400, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "sourceHandle": "text", "target": "n2", "targetHandle": "text"},
        ],
    })
    assert resp.status_code == 201
    wf_id = resp.json()["id"]

    # Update
    resp = await db_client.patch(f"/api/v1/workflows/{wf_id}", json={"name": "updated"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "updated"

    # Publish
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish")
    assert resp.status_code == 200
    instance_id = resp.json()["instance_id"]

    # Verify published status
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.json()["status"] == "published"

    # Verify instance was created
    resp = await db_client.get(f"/api/v1/instances/{instance_id}")
    assert resp.status_code == 200
    assert resp.json()["source_type"] == "workflow"

    # Unpublish
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/unpublish")
    assert resp.status_code == 200

    # Verify draft status
    resp = await db_client.get(f"/api/v1/workflows/{wf_id}")
    assert resp.json()["status"] == "draft"

    # Delete
    resp = await db_client.delete(f"/api/v1/workflows/{wf_id}")
    assert resp.status_code == 204
