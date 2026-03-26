"""LLM service — calls vLLM or any OpenAI-compatible API."""

import logging

import httpx

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=10))
    return _client


async def call_llm(
    prompt: str,
    base_url: str = "http://localhost:8100",
    model: str = "",
    system: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request and return the assistant reply."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    client = _get_client()
    resp = await client.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def call_llm_with_tools(
    messages: list[dict],
    base_url: str = "http://localhost:8100",
    model: str = "",
    api_key: str | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """Call LLM with tool calling support.

    Returns a dict with ``content`` (str) and optionally ``tool_calls`` (list).
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    client = _get_client()
    resp = await client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]

    result: dict = {"content": message.get("content", "") or ""}
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]

    return result
