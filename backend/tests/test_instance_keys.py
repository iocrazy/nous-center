"""Tests for instance API key management and auth."""


async def _create_preset_and_instance(db_client):
    """Helper: create a preset then an instance based on it."""
    resp = await db_client.post("/api/v1/voices", json={
        "name": "testvoice",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": [],
    })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "test-instance",
    })
    assert resp.status_code == 201
    instance = resp.json()
    return preset_id, instance


async def test_create_instance(db_client):
    preset_id, instance = await _create_preset_and_instance(db_client)
    assert instance["source_type"] == "preset"
    assert instance["source_id"] == preset_id
    assert instance["source_name"] == "testvoice"
    assert instance["name"] == "test-instance"
    assert instance["type"] == "tts"
    assert instance["status"] == "active"
    assert instance["endpoint_path"] == f"/v1/instances/{instance['id']}/synthesize"
    assert instance["params_override"] == {}


async def test_list_instances_by_type(db_client):
    preset_id, _ = await _create_preset_and_instance(db_client)

    await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "second-instance",
    })

    # Filter by type
    resp = await db_client.get("/api/v1/instances?type=tts")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # No image instances
    resp = await db_client.get("/api/v1/instances?type=image")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_instance_status_toggle(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    # Toggle to inactive
    resp = await db_client.patch(f"/api/v1/instances/{instance_id}/status", json={
        "status": "inactive",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "inactive"

    # Verify via GET
    resp = await db_client.get(f"/api/v1/instances/{instance_id}")
    assert resp.json()["status"] == "inactive"

    # Toggle back
    resp = await db_client.patch(f"/api/v1/instances/{instance_id}/status", json={
        "status": "active",
    })
    assert resp.json()["status"] == "active"


async def test_delete_instance(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    resp = await db_client.delete(f"/api/v1/instances/{instance_id}")
    assert resp.status_code == 204

    resp = await db_client.get(f"/api/v1/instances/{instance_id}")
    assert resp.status_code == 404


async def test_create_key_and_list(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    # Create an API key
    resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={
        "label": "测试App",
    })
    assert resp.status_code == 201
    key_data = resp.json()
    assert "key" in key_data
    assert key_data["key"].startswith("sk-")
    assert key_data["label"] == "测试App"
    assert key_data["key_prefix"] == key_data["key"][:10]
    assert key_data["is_active"] is True
    assert key_data["usage_calls"] == 0
    assert key_data["usage_chars"] == 0
    key_id = key_data["id"]
    full_key = key_data["key"]

    # List keys — should not include full key
    resp = await db_client.get(f"/api/v1/instances/{instance_id}/keys")
    assert resp.status_code == 200
    keys = resp.json()
    assert len(keys) == 1
    assert keys[0]["id"] == key_id
    assert "key" not in keys[0]
    assert keys[0]["key_prefix"] == full_key[:10]

    # Delete key
    resp = await db_client.delete(f"/api/v1/instances/{instance_id}/keys/{key_id}")
    assert resp.status_code == 204

    # Verify deleted
    resp = await db_client.get(f"/api/v1/instances/{instance_id}/keys")
    assert len(resp.json()) == 0


async def test_create_key_nonexistent_instance(db_client):
    resp = await db_client.post("/api/v1/instances/9999999999999/keys", json={
        "label": "test",
    })
    assert resp.status_code == 404


async def test_bearer_auth_valid_key(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={
        "label": "auth-test-key",
    })
    full_key = resp.json()["key"]

    # Call synthesize — will fail with 409 (engine not loaded)
    # but should NOT fail with 401 (auth should pass)
    resp = await db_client.post(
        f"/v1/instances/{instance_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": f"Bearer {full_key}"},
    )
    assert resp.status_code == 409


async def test_bearer_auth_invalid_key(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    resp = await db_client.post(
        f"/v1/instances/{instance_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": "Bearer sk-fake-invalidkey12345678901234567890"},
    )
    assert resp.status_code == 401


async def test_bearer_auth_no_header(db_client):
    resp = await db_client.post(
        "/v1/instances/12345/synthesize",
        json={"text": "hello"},
    )
    assert resp.status_code == 422


async def test_bearer_auth_inactive_instance(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={
        "label": "will-be-blocked",
    })
    full_key = resp.json()["key"]

    # Deactivate instance
    await db_client.patch(f"/api/v1/instances/{instance_id}/status", json={
        "status": "inactive",
    })

    resp = await db_client.post(
        f"/v1/instances/{instance_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": f"Bearer {full_key}"},
    )
    assert resp.status_code == 403


async def test_multiple_keys_per_instance(db_client):
    _, instance = await _create_preset_and_instance(db_client)
    instance_id = instance["id"]

    keys = []
    for i in range(3):
        resp = await db_client.post(f"/api/v1/instances/{instance_id}/keys", json={
            "label": f"App {i}",
        })
        assert resp.status_code == 201
        keys.append(resp.json()["key"])

    for key in keys:
        resp = await db_client.post(
            f"/v1/instances/{instance_id}/synthesize",
            json={"text": "test"},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert resp.status_code == 409

    resp = await db_client.get(f"/api/v1/instances/{instance_id}/keys")
    assert len(resp.json()) == 3


async def test_instance_params_override(db_client):
    resp = await db_client.post("/api/v1/voices", json={
        "name": "override-test",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "fast-instance",
        "params_override": {"speed": 1.5},
    })
    assert resp.status_code == 201
    assert resp.json()["params_override"] == {"speed": 1.5}

    instance_id = resp.json()["id"]
    resp = await db_client.patch(f"/api/v1/instances/{instance_id}", json={
        "params_override": {"speed": 2.0, "voice": "narrator"},
    })
    assert resp.status_code == 200
    assert resp.json()["params_override"] == {"speed": 2.0, "voice": "narrator"}


async def test_create_instance_with_source_type(db_client):
    """Instance creation uses source_type + source_id instead of preset_id."""
    resp = await db_client.post("/api/v1/voices", json={
        "name": "source-type-test",
        "engine": "cosyvoice2",
        "params": {"voice": "default"},
        "tags": [],
    })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]

    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": preset_id,
        "name": "new-style-instance",
        "type": "tts",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["source_type"] == "preset"
    assert data["source_id"] == preset_id
    assert data["name"] == "new-style-instance"
    assert "preset_id" not in data


async def test_create_instance_invalid_source_type(db_client):
    """Unknown source_type should be rejected by Literal validation."""
    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "unknown",
        "source_id": 12345,
        "name": "bad-instance",
    })
    assert resp.status_code == 422  # Literal["preset", "workflow"] validation


async def test_create_instance_nonexistent_source(db_client):
    """Non-existent source_id should 404."""
    resp = await db_client.post("/api/v1/instances", json={
        "source_type": "preset",
        "source_id": 9999999999999,
        "name": "ghost-instance",
    })
    assert resp.status_code == 404


async def test_list_all_instances(db_client):
    """List without type filter returns all instances."""
    preset_id, _ = await _create_preset_and_instance(db_client)
    resp = await db_client.get("/api/v1/instances")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
