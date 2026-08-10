import pytest
import httpx
import src.api.routes.comfy_templates as ct_mod


class FakeClient:
    async def health(self):
        return {"online": True, "queue_depth": 2, "version": "0.30.2"}
    async def object_info(self):
        return {"RandomNoise": {"input": {"required": {"noise_seed": ["INT", {"default": 0}]}}}}


@pytest.mark.asyncio
async def test_health(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/health")
    assert r.status_code == 200 and r.json()["queue_depth"] == 2


@pytest.mark.asyncio
async def test_object_info_cached(client, monkeypatch):
    calls = {"n": 0}
    class Counting(FakeClient):
        async def object_info(self):
            calls["n"] += 1
            return await super().object_info()
    monkeypatch.setattr(ct_mod, "get_client", lambda: Counting())
    ct_mod._object_info_cache = None
    await client.get("/api/v1/comfy/object-info")
    await client.get("/api/v1/comfy/object-info")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_object_info_sidecar_unreachable(client, monkeypatch):
    class FailingClient(FakeClient):
        async def object_info(self):
            raise httpx.ConnectError("sidecar down")
    monkeypatch.setattr(ct_mod, "get_client", lambda: FailingClient())
    ct_mod._object_info_cache = None
    r = await client.get("/api/v1/comfy/object-info")
    assert r.status_code == 502
