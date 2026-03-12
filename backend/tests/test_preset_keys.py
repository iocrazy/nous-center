"""Tests for preset API key management and auth."""


async def test_create_key_and_list(db_client):
    # Create a preset first
    resp = await db_client.post("/api/v1/voices", json={
        "name": "testvoice",
        "engine": "cosyvoice2",
        "params": {"voice": "default", "speed": 1.0},
        "tags": [],
    })
    assert resp.status_code == 201
    preset_id = resp.json()["id"]

    # Create an API key
    resp = await db_client.post(f"/api/v1/presets/{preset_id}/keys", json={
        "label": "测试App",
    })
    assert resp.status_code == 201
    key_data = resp.json()
    assert "key" in key_data  # Full key only on creation
    assert key_data["key"].startswith("sk-")
    assert key_data["label"] == "测试App"
    assert key_data["key_prefix"] == key_data["key"][:10]
    assert key_data["is_active"] is True
    assert key_data["usage_calls"] == 0
    assert key_data["usage_chars"] == 0
    key_id = key_data["id"]
    full_key = key_data["key"]

    # List keys — should not include full key
    resp = await db_client.get(f"/api/v1/presets/{preset_id}/keys")
    assert resp.status_code == 200
    keys = resp.json()
    assert len(keys) == 1
    assert keys[0]["id"] == key_id
    assert "key" not in keys[0]  # Full key NOT in list response
    assert keys[0]["key_prefix"] == full_key[:10]

    # Delete key
    resp = await db_client.delete(f"/api/v1/presets/{preset_id}/keys/{key_id}")
    assert resp.status_code == 204

    # Verify deleted
    resp = await db_client.get(f"/api/v1/presets/{preset_id}/keys")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_create_key_nonexistent_preset(db_client):
    resp = await db_client.post("/api/v1/presets/9999999999999/keys", json={
        "label": "test",
    })
    assert resp.status_code == 404


async def test_preset_status_toggle(db_client):
    # Create preset
    resp = await db_client.post("/api/v1/voices", json={
        "name": "status-test",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    # Default status is active
    resp = await db_client.get(f"/api/v1/voices/{preset_id}")
    assert resp.json()["status"] == "active"

    # Toggle to inactive
    resp = await db_client.patch(f"/api/v1/presets/{preset_id}/status", json={
        "status": "inactive",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "inactive"

    # Verify via GET
    resp = await db_client.get(f"/api/v1/voices/{preset_id}")
    assert resp.json()["status"] == "inactive"

    # Toggle back to active
    resp = await db_client.patch(f"/api/v1/presets/{preset_id}/status", json={
        "status": "active",
    })
    assert resp.json()["status"] == "active"


async def test_bearer_auth_valid_key(db_client):
    # Create preset + key
    resp = await db_client.post("/api/v1/voices", json={
        "name": "authtest",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    resp = await db_client.post(f"/api/v1/presets/{preset_id}/keys", json={
        "label": "auth-test-key",
    })
    full_key = resp.json()["key"]

    # Call synthesize endpoint — will fail with 409 (engine not loaded)
    # but should NOT fail with 401 (auth should pass)
    resp = await db_client.post(
        f"/v1/preset/{preset_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": f"Bearer {full_key}"},
    )
    # 409 = engine not loaded = auth passed successfully
    assert resp.status_code == 409


async def test_bearer_auth_invalid_key(db_client):
    # Create preset
    resp = await db_client.post("/api/v1/voices", json={
        "name": "authtest2",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    # Call with invalid key
    resp = await db_client.post(
        f"/v1/preset/{preset_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": "Bearer sk-fake-invalidkey12345678901234567890"},
    )
    assert resp.status_code == 401


async def test_bearer_auth_no_header(db_client):
    resp = await db_client.post(
        "/v1/preset/12345/synthesize",
        json={"text": "hello"},
    )
    assert resp.status_code == 422  # Missing required header


async def test_bearer_auth_inactive_preset(db_client):
    # Create preset + key, then deactivate
    resp = await db_client.post("/api/v1/voices", json={
        "name": "inactive-test",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    resp = await db_client.post(f"/api/v1/presets/{preset_id}/keys", json={
        "label": "will-be-blocked",
    })
    full_key = resp.json()["key"]

    # Deactivate preset
    await db_client.patch(f"/api/v1/presets/{preset_id}/status", json={
        "status": "inactive",
    })

    # Try to use key — should get 403
    resp = await db_client.post(
        f"/v1/preset/{preset_id}/synthesize",
        json={"text": "hello"},
        headers={"Authorization": f"Bearer {full_key}"},
    )
    assert resp.status_code == 403


async def test_multiple_keys_per_preset(db_client):
    resp = await db_client.post("/api/v1/voices", json={
        "name": "multikey",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    # Create 3 keys
    keys = []
    for i in range(3):
        resp = await db_client.post(f"/api/v1/presets/{preset_id}/keys", json={
            "label": f"App {i}",
        })
        assert resp.status_code == 201
        keys.append(resp.json()["key"])

    # All should authenticate
    for key in keys:
        resp = await db_client.post(
            f"/v1/preset/{preset_id}/synthesize",
            json={"text": "test"},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert resp.status_code == 409  # Auth passes, engine not loaded

    # List should show 3
    resp = await db_client.get(f"/api/v1/presets/{preset_id}/keys")
    assert len(resp.json()) == 3


async def test_endpoint_path_auto_set(db_client):
    resp = await db_client.post("/api/v1/voices", json={
        "name": "endpoint-test",
        "engine": "cosyvoice2",
        "params": {},
        "tags": [],
    })
    preset_id = resp.json()["id"]

    # Initially no endpoint_path
    resp = await db_client.get(f"/api/v1/voices/{preset_id}")
    assert resp.json()["endpoint_path"] is None

    # Create a key — should auto-set endpoint_path
    await db_client.post(f"/api/v1/presets/{preset_id}/keys", json={
        "label": "trigger-endpoint",
    })

    resp = await db_client.get(f"/api/v1/voices/{preset_id}")
    assert resp.json()["endpoint_path"] == f"/v1/preset/{preset_id}/synthesize"
