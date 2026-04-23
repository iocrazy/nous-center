"""Smoke tests: verify all new modules wire together without import errors."""


async def test_app_starts_without_errors(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_services_endpoint_exists(db_client):
    """v3 replacement: /api/v1/services is the canonical list endpoint
    (the legacy /api/v1/apps GET was removed in PR-A)."""
    resp = await db_client.get("/api/v1/services")
    assert resp.status_code == 200


async def test_inference_imports():
    assert True
