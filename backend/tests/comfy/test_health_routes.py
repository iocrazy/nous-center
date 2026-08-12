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
    monkeypatch.setattr(ct_mod, "_FREE_POLL_INTERVAL", 0)  # 显存不降时别空等 6s
    r = await client.post("/api/v1/comfy/free")
    assert r.status_code == 200
    assert calls["n"] == 1
    body = r.json()
    assert body["ok"] is True
    assert body["devices"][0]["vram_used"] == 102_000_000_000 - 37_700_000_000
    # 显存没降 → settled False,前端据此提示"释放已触发,可能仍在进行"
    assert body["settled"] is False


@pytest.mark.asyncio
async def test_free_waits_for_async_unload_to_land(client, monkeypatch):
    """ComfyUI 的 /free 只设 flag,真卸载晚几秒才发生(实测 64.3G→43.8G)。
    路由要轮询到显存真降下来再返回,否则按钮看着像没生效。"""
    state = {"round": 0}

    class Lagging(FakeClient):
        async def free(self):
            return None

        async def system_stats(self):
            # 第 3 次查询才反映卸载完成(模拟 worker 被唤醒后的延迟)
            state["round"] += 1
            free_b = 37_700_000_000 if state["round"] < 3 else 81_000_000_000
            return {"devices": [{"name": "cuda:0 Test GPU", "type": "cuda", "index": 0,
                                 "vram_total": 102_000_000_000, "vram_free": free_b,
                                 "torch_vram_total": 0}]}

    monkeypatch.setattr(ct_mod, "get_client", lambda: Lagging())
    monkeypatch.setattr(ct_mod, "_FREE_POLL_INTERVAL", 0)
    body = (await client.post("/api/v1/comfy/free")).json()
    assert body["settled"] is True
    assert body["freed_bytes"] > 1_000_000_000
    assert body["devices"][0]["vram_used"] == 102_000_000_000 - 81_000_000_000


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


@pytest.mark.asyncio
async def test_free_noop_when_nothing_resident(client, monkeypatch):
    """空载时(只剩 ~1.6G torch 上下文)点释放:落定且归还 0,UI 说"没有可释放的",
    不能报"仍在卸载"误导用户(实机验证发现)。"""
    class Idle(FakeClient):
        async def free(self):
            return None

        async def system_stats(self):
            return {"devices": [{"name": "cuda:0 Test GPU", "type": "cuda", "index": 0,
                                 "vram_total": 102_000_000_000,
                                 "vram_free": 100_400_000_000, "torch_vram_total": 0}]}

    monkeypatch.setattr(ct_mod, "get_client", lambda: Idle())
    monkeypatch.setattr(ct_mod, "_FREE_POLL_INTERVAL", 0)
    body = (await client.post("/api/v1/comfy/free")).json()
    assert body["settled"] is True and body["freed_bytes"] == 0
