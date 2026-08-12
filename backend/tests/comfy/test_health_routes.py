import pytest
import httpx
import src.api.routes.comfy_templates as ct_mod


DEVICE = {
    "name": "cuda:0 NVIDIA RTX PRO 6000 Blackwell Workstation Edition : cudaMallocAsync",
    "type": "cuda",
    "index": 0,
    "vram_total": 102_000_000_000,
    "vram_free": 37_700_000_000,
    "torch_vram_total": 60_000_000_000,
}


class FakeClient:
    async def health(self):
        return {"online": True, "queue_depth": 2, "version": "0.30.2"}
    async def object_info(self):
        return {"RandomNoise": {"input": {"required": {"noise_seed": ["INT", {"default": 0}]}}}}
    async def system_stats(self):
        return {"devices": [DEVICE]}
    async def free(self):
        return None


@pytest.mark.asyncio
async def test_health(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/health")
    assert r.status_code == 200 and r.json()["queue_depth"] == 2


@pytest.mark.asyncio
async def test_health_includes_devices_with_computed_vram_used(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/health")
    body = r.json()
    assert body["devices"] == [{
        "name": DEVICE["name"],
        "type": "cuda",
        "index": 0,
        "vram_total": 102_000_000_000,
        "vram_free": 37_700_000_000,
        "vram_used": 102_000_000_000 - 37_700_000_000,
        "torch_vram_total": 60_000_000_000,
    }]


@pytest.mark.asyncio
async def test_health_offline_devices_empty(client, monkeypatch):
    class OfflineClient(FakeClient):
        async def health(self):
            return {"online": False, "queue_depth": 0, "version": ""}
    monkeypatch.setattr(ct_mod, "get_client", lambda: OfflineClient())
    r = await client.get("/api/v1/comfy/health")
    assert r.status_code == 200
    assert r.json()["devices"] == []


@pytest.mark.asyncio
async def test_free_calls_client_and_returns_snapshot(client, monkeypatch):
    calls = {"n": 0}
    class Counting(FakeClient):
        async def free(self):
            calls["n"] += 1
    monkeypatch.setattr(ct_mod, "get_client", lambda: Counting())
    r = await client.post("/api/v1/comfy/free")
    assert r.status_code == 200
    assert calls["n"] == 1
    body = r.json()
    assert body["ok"] is True
    assert body["devices"][0]["vram_used"] == 102_000_000_000 - 37_700_000_000


@pytest.mark.asyncio
async def test_free_propagates_comfy_error(client, monkeypatch):
    from src.services.comfy.client import ComfyError

    class FailingClient(FakeClient):
        async def free(self):
            raise ComfyError("释放显存失败(HTTP 500)")
    monkeypatch.setattr(ct_mod, "get_client", lambda: FailingClient())
    r = await client.post("/api/v1/comfy/free")
    assert r.status_code == 502


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
