"""Tests for LLM service."""

import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.llm_service import call_llm


def _mock_response(content: str = "Hello from LLM") -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    return resp


@pytest.mark.asyncio
async def test_call_llm_basic():
    mock_post = AsyncMock(return_value=_mock_response("reply text"))

    with patch("src.services.llm_service.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = mock_post
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = client_instance

        result = await call_llm(prompt="hi", base_url="http://test:8100", model="m1")

    assert result == "reply text"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["model"] == "m1"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_call_llm_with_system_message():
    mock_post = AsyncMock(return_value=_mock_response("ok"))

    with patch("src.services.llm_service.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = mock_post
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = client_instance

        await call_llm(prompt="hello", system="You are helpful.")

    body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert body["messages"][1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_call_llm_with_api_key():
    mock_post = AsyncMock(return_value=_mock_response("ok"))

    with patch("src.services.llm_service.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = mock_post
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = client_instance

        await call_llm(prompt="hi", api_key="sk-test")

    headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers")
    assert headers["Authorization"] == "Bearer sk-test"
