"""VLLMAdapter — manages vLLM as a subprocess with full lifecycle control."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
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
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        quantization: str | None = None,
        dtype: str | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, device=device)
        self._port = vllm_port or (int(vllm_base_url.split(":")[-1]) if vllm_base_url else 0)
        self._tp = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_mem_util = gpu_memory_utilization
        self._quantization = quantization
        self._max_num_seqs = max_num_seqs
        self._dtype = dtype
        # Port resolved lazily in load() if not set
        if self._port:
            self._base_url = f"http://localhost:{self._port}"
        else:
            self._base_url = None  # Will be set in load()
        self.base_url = self._base_url
        self._process: subprocess.Popen | None = None
        self._client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=10))
        self._managed = True  # True = we control the subprocess

    def _auto_configure(self, device: str | None) -> dict:
        """Auto-calculate vLLM launch parameters based on model and GPU state."""
        import json
        from src.config import get_settings
        from src.services.gpu_monitor import poll_gpu_stats

        settings = get_settings()
        model_path = Path(settings.LOCAL_MODELS_PATH) / self.model_path
        if not model_path.exists():
            model_path = Path(self.model_path)

        # 1. Read model config.json
        config_file = model_path / "config.json"
        model_config: dict = {}
        if config_file.exists():
            with open(config_file) as f:
                model_config = json.load(f)

        # 2. Get model size from safetensors / bin files
        model_size_gb = sum(
            f.stat().st_size for f in model_path.glob("*.safetensors")
        ) / (1024**3)
        if model_size_gb == 0:
            model_size_gb = sum(
                f.stat().st_size for f in model_path.glob("*.bin")
            ) / (1024**3)

        # 3. Auto-detect quantization from config
        quant_config = model_config.get("quantization_config", {})
        quantization = quant_config.get("quant_method")  # "gptq", "awq", "compressed-tensors", etc

        # 4. Auto-detect dtype
        dtype = None
        if quantization in ("gptq",):
            dtype = "float16"  # GPTQ requires float16

        # 5. Get GPU info
        gpu_stats = poll_gpu_stats()

        # 6. Determine GPU index
        if device and ":" in device:
            gpu_idx = int(device.split(":")[-1])
        else:
            gpu_idx = (
                max(range(len(gpu_stats)), key=lambda i: gpu_stats[i]["free_mb"])
                if gpu_stats
                else 0
            )

        gpu_total_gb = gpu_stats[gpu_idx]["total_mb"] / 1024 if gpu_idx < len(gpu_stats) else 24.0
        gpu_free_gb = gpu_stats[gpu_idx]["free_mb"] / 1024 if gpu_idx < len(gpu_stats) else 24.0

        # 7. Determine tensor_parallel_size
        tp = self._tp or 1
        if tp <= 1 and model_size_gb > gpu_free_gb * 0.85:
            total_free = sum(g["free_mb"] for g in gpu_stats) / 1024
            if total_free > model_size_gb * 1.2:
                tp = len(gpu_stats)
            else:
                tp = 1

        # 8. Calculate gpu_memory_utilization
        if tp > 1:
            per_gpu_model = model_size_gb / tp
            kv_buffer_gb = 4.0
            needed = per_gpu_model + kv_buffer_gb
            utilization = min(0.85, needed / gpu_total_gb)
        else:
            kv_buffer_gb = min(4.0, gpu_free_gb - model_size_gb - 1.0)
            if kv_buffer_gb < 1.0:
                kv_buffer_gb = 1.0
            needed = model_size_gb + kv_buffer_gb + 1.0  # +1GB CUDA overhead
            utilization = min(0.92, needed / gpu_total_gb)

        # 9. Calculate max_model_len
        max_position = model_config.get("max_position_embeddings", 131072)
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_kv_heads = model_config.get("num_key_value_heads", 8)
        head_dim = hidden_size // model_config.get("num_attention_heads", 32)
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # key+value, fp16
        kv_cache_bytes = kv_buffer_gb * 1024 * 1024 * 1024
        estimated_max_tokens = int(kv_cache_bytes / bytes_per_token) if bytes_per_token > 0 else 4096
        max_model_len = min(max_position, max(1024, estimated_max_tokens))
        max_model_len = (max_model_len // 1024) * 1024  # round down to nearest 1024

        # 10. Calculate max_num_seqs
        max_num_seqs = min(256, max(8, estimated_max_tokens // 128))

        # 11. Find free port
        port = self._port
        if not port:
            import socket
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        return {
            "port": port,
            "tp": tp,
            "max_model_len": max_model_len,
            "utilization": round(utilization, 2),
            "quantization": quantization,
            "dtype": dtype,
            "max_num_seqs": max_num_seqs,
            "gpu_idx": gpu_idx,
            "model_size_gb": round(model_size_gb, 2),
        }

    async def load(self, device: str | None = None) -> None:
        """Start vLLM subprocess or connect to existing instance."""
        # First check if vLLM is already running on this port
        if self._base_url and await self._health_check():
            self._model = True
            self._managed = False  # We didn't start it, don't kill it
            logger.info("Connected to existing vLLM at %s", self._base_url)
            return

        # Auto-configure parameters
        auto = self._auto_configure(device)
        port = self._port or auto["port"]
        tp = self._tp or auto["tp"]
        max_model_len = self._max_model_len or auto["max_model_len"]
        utilization = self._gpu_mem_util or auto["utilization"]
        quantization = self._quantization or auto["quantization"]
        dtype = self._dtype or auto["dtype"]
        max_num_seqs = self._max_num_seqs or auto["max_num_seqs"]

        # Update base_url now that port is resolved
        self._port = port
        self._base_url = f"http://localhost:{port}"
        self.base_url = self._base_url

        logger.info(
            "Auto-config: model=%.1fGB, tp=%d, max_len=%d, util=%.2f, seqs=%d, quant=%s",
            auto["model_size_gb"], tp, max_model_len, utilization, max_num_seqs, quantization,
        )

        # Resolve model path
        from src.config import get_settings
        settings = get_settings()
        model_path = str(Path(settings.LOCAL_MODELS_PATH) / self.model_path)
        if not Path(model_path).exists():
            model_path = str(self.model_path)  # Try as absolute path

        # Build vLLM command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(utilization),
        ]
        if tp > 1:
            cmd += ["--tensor-parallel-size", str(tp)]
        if quantization:
            cmd += ["--quantization", quantization]
        if dtype:
            cmd += ["--dtype", dtype]
        if max_num_seqs:
            cmd += ["--max-num-seqs", str(max_num_seqs)]

        # Set CUDA_VISIBLE_DEVICES for single-GPU mode
        env = dict(os.environ)
        if tp <= 1 and device:
            # Extract GPU index from device string like "cuda:0"
            gpu_idx = device.split(":")[-1] if ":" in device else "0"
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
            logger.info("Starting vLLM on GPU %s: %s", gpu_idx, " ".join(cmd))
        else:
            logger.info("Starting vLLM (TP=%d): %s", tp, " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
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
