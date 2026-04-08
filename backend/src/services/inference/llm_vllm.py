"""VLLMAdapter — manages vLLM as a subprocess with full lifecycle control."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from src.services.inference.base import InferenceAdapter, InferenceResult

logger = logging.getLogger(__name__)

# Port counter for multiple vLLM instances
_next_port = 8100


def _alloc_port() -> int:
    global _next_port
    port = _next_port
    _next_port += 1
    return port


class VLLMAdapter(InferenceAdapter):
    """Adapter that spawns vLLM as a subprocess and manages its lifecycle.

    load()   → start vLLM subprocess → wait for health check
    unload() → kill subprocess → free GPU memory
    infer()  → HTTP call to the local vLLM instance
    """

    model_type = "llm"
    estimated_vram_mb = 0  # Determined at runtime

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        vllm_base_url: str | None = None,
        vllm_port: int | None = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        quantization: str | None = None,
        dtype: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, device=device)
        self._port = vllm_port or (int(vllm_base_url.split(":")[-1]) if vllm_base_url else _alloc_port())
        self._tp = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_mem_util = gpu_memory_utilization
        self._quantization = quantization
        self._dtype = dtype
        self._base_url = f"http://localhost:{self._port}"
        self.base_url = self._base_url
        self._process: subprocess.Popen | None = None
        self._client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=10))
        self._managed = True  # True = we control the subprocess

    async def load(self, device: str | None = None) -> None:
        """Start vLLM subprocess or connect to existing instance."""
        # First check if vLLM is already running on this port
        if await self._health_check():
            self._model = True
            self._managed = False  # We didn't start it, don't kill it
            logger.info("Connected to existing vLLM at %s", self._base_url)
            return

        # Resolve model path
        from src.config import get_settings
        settings = get_settings()
        model_path = str(Path(settings.LOCAL_MODELS_PATH) / self.model_path)
        if not Path(model_path).exists():
            model_path = str(self.model_path)  # Try as absolute path

        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(self._port),
            "--max-model-len", str(self._max_model_len),
            "--gpu-memory-utilization", str(self._gpu_mem_util),
        ]
        if self._tp > 1:
            cmd += ["--tensor-parallel-size", str(self._tp)]
        if self._quantization:
            cmd += ["--quantization", self._quantization]
        if self._dtype:
            cmd += ["--dtype", self._dtype]

        logger.info("Starting vLLM: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._managed = True

        # Wait for vLLM to become healthy (up to 5 minutes)
        start = time.monotonic()
        timeout = 300
        while time.monotonic() - start < timeout:
            if self._process.poll() is not None:
                # Process exited
                output = self._process.stdout.read() if self._process.stdout else ""
                logger.error("vLLM exited with code %d: %s", self._process.returncode, output[-500:])
                self._process = None
                raise RuntimeError(f"vLLM failed to start: {output[-200:]}")

            if await self._health_check():
                elapsed = int(time.monotonic() - start)
                self._model = True
                logger.info("vLLM ready in %ds at %s", elapsed, self._base_url)
                return

            await asyncio.sleep(5)

        # Timeout
        self._kill_process()
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")

    def unload(self) -> None:
        """Kill vLLM subprocess and release GPU memory."""
        if self._managed and self._process is not None:
            self._kill_process()
            logger.info("vLLM subprocess killed, GPU memory released")
        else:
            logger.info("vLLM was external, disconnecting only")
        self._model = None

    def _kill_process(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.warning("Error killing vLLM: %s", e)
        finally:
            self._process = None

    async def _health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    async def infer(self, params: dict[str, Any]) -> InferenceResult:
        """Forward chat completion request to vLLM."""
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions", json=params
        )
        return InferenceResult(data=resp.content, content_type="application/json")

    async def infer_stream(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream SSE chunks from vLLM."""
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            json={**params, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line.encode() + b"\n"
