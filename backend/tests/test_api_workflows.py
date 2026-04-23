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


# Legacy publish/unpublish/run lifecycle tests were removed in PR-A
# (v3 IA rebuild). The legacy POST /api/v1/workflows/{id}/publish handler
# was deleted; the v3 publish path is covered by tests/test_workflow_publish.py
# and the run path will land with PR-B's frontend rewiring.
