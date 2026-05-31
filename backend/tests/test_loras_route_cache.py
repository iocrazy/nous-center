"""round7:/api/v1/loras 走 @cached → 必须带 ETag(早先 handler 无 request 参数,
@cached 取不到 Request 直接 fall-through 不缓存)。"""
import pytest


@pytest.mark.asyncio
async def test_loras_route_returns_etag(client):
    r = await client.get("/api/v1/loras")
    assert r.status_code == 200
    # @cached 生效 → 有 ETag(早先无 request 参数时这个头不存在)
    assert "etag" in {k.lower() for k in r.headers}


@pytest.mark.asyncio
async def test_loras_route_304_on_matching_etag(client):
    r1 = await client.get("/api/v1/loras")
    etag = r1.headers.get("etag")
    assert etag
    r2 = await client.get("/api/v1/loras", headers={"If-None-Match": etag})
    assert r2.status_code == 304
