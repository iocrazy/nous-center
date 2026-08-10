import json
import httpx
import pytest

from src.services.comfy.client import ComfyClient, ComfyError, translate_prompt_error


def make_client(handler) -> ComfyClient:
    c = ComfyClient(base_url="http://comfy.test")
    c._client = httpx.AsyncClient(  # noqa: SLF001 — 测试注入 MockTransport
        transport=httpx.MockTransport(handler), base_url="http://comfy.test")
    return c


@pytest.mark.asyncio
async def test_submit_returns_prompt_id():
    def handler(req):
        assert req.url.path == "/prompt"
        return httpx.Response(200, json={"prompt_id": "abc123"})
    assert await make_client(handler).submit({"1": {"class_type": "X", "inputs": {}}}) == "abc123"


@pytest.mark.asyncio
async def test_submit_validation_error_translated():
    def handler(req):
        return httpx.Response(400, json={"error": {"message": "invalid prompt"}, "node_errors": {
            "138": {"errors": [{"message": "Required input is missing", "extra_info": {"input_name": "text"}}],
                     "class_type": "PrimitiveStringMultiline"}}})
    with pytest.raises(ComfyError) as e:
        await make_client(handler).submit({})
    assert "138" in e.value.message and "text" in e.value.message


@pytest.mark.asyncio
async def test_wait_polls_until_history_present():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        body = {} if calls["n"] < 3 else {"p1": {"outputs": {"92": {"images": []}}}}
        return httpx.Response(200, json=body)
    hist = await make_client(handler).wait("p1", timeout_s=10, interval_s=0)
    assert "outputs" in hist and calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_timeout_raises():
    def handler(req):
        return httpx.Response(200, json={})
    with pytest.raises(ComfyError, match="超时"):
        await make_client(handler).wait("p1", timeout_s=0.01, interval_s=0)


@pytest.mark.asyncio
async def test_health_offline_on_connect_error():
    def handler(req):
        raise httpx.ConnectError("refused")
    h = await make_client(handler).health()
    assert h["online"] is False


def test_translate_prompt_error_fallback_plain_text():
    msg = translate_prompt_error(500, "Internal Server Error")
    assert "HTTP 500" in msg
