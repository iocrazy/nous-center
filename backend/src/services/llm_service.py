"""LLM service — calls vLLM or any OpenAI-compatible API."""

import logging

import httpx

logger = logging.getLogger(__name__)


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

    async with httpx.AsyncClient(timeout=120) as client:
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
