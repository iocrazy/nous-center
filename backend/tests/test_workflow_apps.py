import pytest


async def test_publish_creates_app(db_client):
    resp = await db_client.post("/api/v1/workflows", json={
        "name": "test-wf", "nodes": [{"id": "n1", "type": "text_input", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}}], "edges": [],
    }, headers={"X-Admin-Token": ""})
    assert resp.status_code == 201
    wf_id = resp.json()["id"]
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish-app", json={
        "name": "my-test-app", "display_name": "Test App", "description": "A test app",
        "exposed_inputs": [{"node_id": "n1", "param_key": "text", "api_name": "text", "param_type": "string", "description": "Input text", "required": True, "default": None}],
        "exposed_outputs": [],
    })
    assert resp.status_code == 201
    assert resp.json()["name"] == "my-test-app"
    assert resp.json()["active"] is True


async def test_list_apps(db_client):
    resp = await db_client.get("/api/v1/apps")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_unpublish_app(db_client):
    resp = await db_client.post("/api/v1/workflows", json={"name": "wf2", "nodes": [], "edges": []}, headers={"X-Admin-Token": ""})
    wf_id = resp.json()["id"]
    await db_client.post(f"/api/v1/workflows/{wf_id}/publish-app", json={
        "name": "app-to-delete", "display_name": "Del", "description": "", "exposed_inputs": [], "exposed_outputs": [],
    })
    resp = await db_client.delete("/api/v1/apps/app-to-delete")
    assert resp.status_code == 204


async def test_duplicate_name_rejected(db_client):
    resp = await db_client.post("/api/v1/workflows", json={"name": "wf3", "nodes": [], "edges": []}, headers={"X-Admin-Token": ""})
    wf_id = resp.json()["id"]
    await db_client.post(f"/api/v1/workflows/{wf_id}/publish-app", json={
        "name": "unique-app", "display_name": "U", "description": "", "exposed_inputs": [], "exposed_outputs": [],
    })
    resp = await db_client.post(f"/api/v1/workflows/{wf_id}/publish-app", json={
        "name": "unique-app", "display_name": "U2", "description": "", "exposed_inputs": [], "exposed_outputs": [],
    })
    assert resp.status_code == 409
