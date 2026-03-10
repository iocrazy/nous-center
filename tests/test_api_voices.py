async def test_create_and_list_presets(db_client):
    client = db_client
    # Create
    resp = await client.post("/api/v1/voices", json={
        "name": "test-voice",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": ["test"],
    })
    assert resp.status_code == 201
    preset = resp.json()
    assert preset["name"] == "test-voice"
    preset_id = preset["id"]

    # List
    resp = await client.get("/api/v1/voices")
    assert resp.status_code == 200
    presets = resp.json()
    assert any(p["id"] == preset_id for p in presets)

    # Get by ID
    resp = await client.get(f"/api/v1/voices/{preset_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "test-voice"

    # Update
    resp = await client.put(f"/api/v1/voices/{preset_id}", json={"name": "renamed"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"

    # Delete
    resp = await client.delete(f"/api/v1/voices/{preset_id}")
    assert resp.status_code == 204


async def test_get_nonexistent_preset(db_client):
    resp = await db_client.get("/api/v1/voices/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


async def test_create_and_list_groups(db_client):
    client = db_client
    # Create
    resp = await client.post("/api/v1/voices/groups", json={
        "name": "podcast-cast",
        "presets": ["host-voice", "guest-voice"],
    })
    assert resp.status_code == 201
    group = resp.json()
    assert group["name"] == "podcast-cast"
    assert group["presets"] == ["host-voice", "guest-voice"]
    group_id = group["id"]

    # List
    resp = await client.get("/api/v1/voices/groups")
    assert resp.status_code == 200
    groups = resp.json()
    assert any(g["id"] == group_id for g in groups)

    # Delete
    resp = await client.delete(f"/api/v1/voices/groups/{group_id}")
    assert resp.status_code == 204
