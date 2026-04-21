"""Integration tests: /v1/responses with agent field."""

import pytest
from httpx import AsyncClient


async def test_first_request_with_agent_writes_agent_id(
    monkeypatch, api_client: AsyncClient, bearer_headers, fixtures_home, mock_vllm
):
    """First request with agent=tutor writes agent_id into response_sessions."""
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    resp = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "你好", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["id"].startswith("resp-")
    # 查 DB
    from sqlalchemy import select
    from src.models.response_session import ResponseSession
    async with api_client.app.state.async_session_factory() as db:
        sess = (await db.execute(select(ResponseSession))).scalar_one()
        assert sess.agent_id == "tutor"


async def test_continuation_without_agent_restores_from_session(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    r1 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert r1.status_code == 200, r1.text
    first_id = r1.json()["id"]
    r2 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "again", "previous_response_id": first_id},
        headers=bearer_headers,
    )
    assert r2.status_code == 200, r2.text
    # assert mock_vllm 收到的 messages 首条 system message 包含 tutor 的 IDENTITY
    sent = mock_vllm.last_request_body["messages"]
    assert sent[0]["role"] == "system"
    assert "Tutor" in sent[0]["content"]


async def test_continuation_with_mismatched_agent_returns_400(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    r1 = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert r1.status_code == 200, r1.text
    first_id = r1.json()["id"]
    r2 = await api_client.post(
        "/v1/responses",
        json={
            "model": "qwen3.5",
            "input": "again",
            "previous_response_id": first_id,
            "agent": "writer",
        },
        headers=bearer_headers,
    )
    assert r2.status_code == 400
    # The app's global HTTPException handler flattens detail into
    # error.message (see src/api/main.py::_http). Assert the marker text
    # survives the flattening.
    body = r2.json()
    assert "agent_session_mismatch" in body["error"]["message"]


async def test_messages_order_regression_no_agent(
    monkeypatch, api_client, bearer_headers, mock_vllm
):
    """Without agent, messages array sent to vLLM must be byte-identical to pre-change."""
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    resp = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "instructions": "be brief"},
        headers=bearer_headers,
    )
    assert resp.status_code == 200, resp.text
    sent = mock_vllm.last_request_body["messages"]
    # 第一条应是 instructions 的 system message（不是 agent system message）
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "be brief"


async def test_flag_off_ignores_agent_field(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "false")
    from src.config import get_settings
    get_settings.cache_clear()
    resp = await api_client.post(
        "/v1/responses",
        json={"model": "qwen3.5", "input": "hi", "agent": "tutor"},
        headers=bearer_headers,
    )
    assert resp.status_code == 200, resp.text
    # agent 字段被忽略，session.agent_id 应为 NULL
    from sqlalchemy import select
    from src.models.response_session import ResponseSession
    async with api_client.app.state.async_session_factory() as db:
        sess = (await db.execute(select(ResponseSession))).scalar_one()
        assert sess.agent_id is None


@pytest.mark.asyncio
async def test_chat_completions_with_agent_injects_system_message(
    monkeypatch, api_client, bearer_headers, fixtures_home, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    resp = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5",
            "agent": "tutor",
            "messages": [{"role": "user", "content": "你好"}],
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    sent = mock_vllm.last_request_body["messages"]
    # 首条应是 agent system message
    assert sent[0]["role"] == "system"
    assert "你是 Tutor" in sent[0]["content"]


@pytest.mark.asyncio
async def test_chat_completions_no_agent_unchanged(
    monkeypatch, api_client, bearer_headers, mock_vllm
):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    from src.config import get_settings
    get_settings.cache_clear()
    resp = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=bearer_headers,
    )
    assert resp.status_code == 200
    sent = mock_vllm.last_request_body["messages"]
    assert sent == [{"role": "user", "content": "hi"}]
