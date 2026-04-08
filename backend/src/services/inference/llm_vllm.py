from __future__ import annotations
import logging
from typing import Any, AsyncIterator
import httpx
from src.services.inference.base import InferenceAdapter, InferenceResult

logger = logging.getLogger(__name__)

class VLLMAdapter(InferenceAdapter):
    model_type = "llm"
    estimated_vram_mb = 0

    def __init__(self, model_path: str, device: str = "cuda", vllm_base_url: str = "http://localhost:8100", **kwargs: Any):
        super().__init__(model_path=model_path, device=device)
        self._base_url = vllm_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=10))

    async def load(self, device: str | None = None) -> None:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            if resp.status_code == 200:
                self._model = True
                logger.info("VLLMAdapter connected to %s", self._base_url)
            else:
                self._model = None
        except httpx.ConnectError:
            self._model = None
            logger.warning("Cannot connect to vLLM at %s", self._base_url)

    def unload(self) -> None:
        self._model = None

    async def infer(self, params: dict[str, Any]) -> InferenceResult:
        resp = await self._client.post(f"{self._base_url}/v1/chat/completions", json=params)
        return InferenceResult(data=resp.content, content_type="application/json")

    async def infer_stream(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        async with self._client.stream("POST", f"{self._base_url}/v1/chat/completions",
                                        json={**params, "stream": True}) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line.encode() + b"\n"
