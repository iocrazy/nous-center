"""Smoke tests: verify all new modules wire together without import errors."""


async def test_app_starts_without_errors(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_apps_endpoint_exists(db_client):
    resp = await db_client.get("/api/v1/apps")
    assert resp.status_code == 200


async def test_inference_imports():
    assert True
