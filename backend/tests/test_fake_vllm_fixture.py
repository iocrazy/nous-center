"""Lane J: fake_vllm fixture self-test — mock vLLM HTTP endpoint (chat/completions + health + abort)."""
import asyncio

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fake_vllm_chat_completions(fake_vllm):
    """fake_vllm exposes /v1/chat/completions and returns OpenAI-shaped response."""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        resp = await c.post(
            "/v1/chat/completions",
            json={
                "model": "fake-llm",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert "usage" in body


@pytest.mark.asyncio
async def test_fake_vllm_health(fake_vllm):
    """/health returns 200 — used by LLM runner preload health checks."""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fake_vllm_streaming(fake_vllm):
    """stream=true returns SSE chunks ending in [DONE]."""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "fake-llm",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            chunks = [line async for line in resp.aiter_lines() if line.strip()]
    assert len(chunks) >= 1
    assert any("[DONE]" in c or "finish_reason" in c for c in chunks)


@pytest.mark.asyncio
async def test_fake_vllm_records_concurrent_requests(fake_vllm):
    """fake_vllm records concurrent in-flight requests — used by the
    LLM-not-serialized integration case to assert observed parallelism.
    """

    async def _one():
        async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
            await c.post(
                "/v1/chat/completions",
                json={
                    "model": "fake-llm",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )

    await asyncio.gather(*[_one() for _ in range(4)])
    assert fake_vllm.max_concurrent_seen >= 2


@pytest.mark.asyncio
async def test_fake_vllm_abort(fake_vllm):
    """/v1/abort records the request_id for later assertion."""
    async with httpx.AsyncClient(base_url=fake_vllm.base_url) as c:
        resp = await c.post("/v1/abort", json={"request_id": "req-42"})
    assert resp.status_code == 200
    assert "req-42" in fake_vllm.aborted_ids
