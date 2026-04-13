"""SGLangAdapter — manages SGLang as a subprocess with full lifecycle control.

Drop-in replacement for VLLMAdapter. SGLang serves an OpenAI-compatible API
and is 3-5x faster than vLLM on Qwen3.5 MoE models.
"""
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


class SGLangAdapter(InferenceAdapter):
    """Adapter that spawns SGLang as a subprocess and manages its lifecycle."""

    model_type = "llm"
    estimated_vram_mb = 0

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sglang_port: int | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        quantization: str | None = None,
        dtype: str | None = None,
        max_num_seqs: int | None = None,
        adopt_pid: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_path=model_path, device=device)
        self._port = sglang_port or 0
        self._tp = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_mem_util = gpu_memory_utilization
        self._quantization = quantization
        self._max_num_seqs = max_num_seqs
        self._dtype = dtype
        if self._port:
            self._base_url = f"http://localhost:{self._port}"
        else:
            self._base_url = None
        self.base_url = self._base_url
        self._process: subprocess.Popen | None = None
        self._adopt_pid = adopt_pid
        self._adopted_pid: int | None = None
        self._client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=10), proxy=None)
        self._managed = True

    def _auto_configure(self, device: str | None) -> dict:
        """Auto-calculate SGLang launch parameters based on model and GPU state."""
        import json
        from src.config import get_settings
        from src.services.gpu_monitor import poll_gpu_stats

        settings = get_settings()
        model_path = Path(settings.LOCAL_MODELS_PATH) / self.model_path
        if not model_path.exists():
            model_path = Path(self.model_path)

        config_file = model_path / "config.json"
        model_config: dict = {}
        if config_file.exists():
            with open(config_file) as f:
                model_config = json.load(f)

        model_size_gb = sum(
            f.stat().st_size for f in model_path.glob("*.safetensors")
        ) / (1024**3)
        if model_size_gb == 0:
            model_size_gb = sum(
                f.stat().st_size for f in model_path.glob("*.bin")
            ) / (1024**3)

        quant_config = model_config.get("quantization_config", {})
        quantization = quant_config.get("quant_method")
        # SGLang uses gptq_marlin automatically, just pass gptq
        dtype = None

        gpu_stats = poll_gpu_stats()

        if device and ":" in device:
            gpu_idx = int(device.split(":")[-1])
        else:
            gpu_idx = (
                max(range(len(gpu_stats)), key=lambda i: gpu_stats[i]["free_mb"])
                if gpu_stats else 0
            )

        gpu_total_gb = gpu_stats[gpu_idx]["total_mb"] / 1024 if gpu_idx < len(gpu_stats) else 24.0
        gpu_free_gb = gpu_stats[gpu_idx]["free_mb"] / 1024 if gpu_idx < len(gpu_stats) else 24.0

        tp = self._tp or 1
        if tp <= 1 and model_size_gb > gpu_free_gb * 0.85:
            total_free = sum(g["free_mb"] for g in gpu_stats) / 1024
            if total_free > model_size_gb * 1.2:
                tp = len(gpu_stats)

        if tp > 1:
            per_gpu_model = model_size_gb / tp
            kv_buffer_gb = 4.0
            needed = per_gpu_model + kv_buffer_gb
            utilization = min(0.85, needed / gpu_total_gb)
        else:
            kv_buffer_gb = min(4.0, gpu_free_gb - model_size_gb - 1.0)
            if kv_buffer_gb < 1.0:
                kv_buffer_gb = 1.0
            needed = model_size_gb + kv_buffer_gb + 1.0
            utilization = min(0.92, needed / gpu_total_gb)

        if kv_buffer_gb < 3.0:
            max_model_len = 2048
        elif kv_buffer_gb < 6.0:
            max_model_len = 4096
        else:
            max_model_len = 8192

        if kv_buffer_gb < 3.0:
            max_num_seqs = 16
        elif kv_buffer_gb < 6.0:
            max_num_seqs = 32
        else:
            max_num_seqs = 64

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
        """Start SGLang subprocess or connect to existing instance."""
        if self._base_url and await self._health_check():
            self._model = True
            if self._adopt_pid:
                self._managed = True
                self._adopted_pid = self._adopt_pid
                logger.info("Adopted orphan SGLang (pid=%d) at %s", self._adopt_pid, self._base_url)
            else:
                self._managed = False
                logger.info("Connected to existing SGLang at %s", self._base_url)
            return

        auto = self._auto_configure(device)
        port = self._port or auto["port"]
        tp = self._tp or auto["tp"]
        max_model_len = self._max_model_len or auto["max_model_len"]
        utilization = self._gpu_mem_util or auto["utilization"]
        quantization = self._quantization or auto["quantization"]
        max_num_seqs = self._max_num_seqs or auto["max_num_seqs"]

        self._port = port
        self._base_url = f"http://localhost:{port}"
        self.base_url = self._base_url

        logger.info(
            "SGLang auto-config: model=%.1fGB, tp=%d, max_len=%d, util=%.2f, seqs=%d, quant=%s",
            auto["model_size_gb"], tp, max_model_len, utilization, max_num_seqs, quantization,
        )

        from src.config import get_settings
        settings = get_settings()
        model_path = str(Path(settings.LOCAL_MODELS_PATH) / self.model_path)
        if not Path(model_path).exists():
            model_path = str(self.model_path)

        # Build SGLang command
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--context-length", str(max_model_len),
            "--mem-fraction-static", str(utilization),
        ]
        if tp > 1:
            cmd += ["--tp", str(tp)]
        if quantization:
            cmd += ["--quantization", quantization]
        if max_num_seqs:
            cmd += ["--max-running-requests", str(max_num_seqs)]

        # Environment
        env = dict(os.environ)
        env["NO_PROXY"] = "localhost,127.0.0.1"
        _cache_root = str(Path(settings.LOCAL_MODELS_PATH) / ".cache")
        env["TORCH_HOME"] = str(Path(_cache_root) / "torch")
        env["XDG_CACHE_HOME"] = _cache_root

        if tp <= 1 and device:
            gpu_idx = device.split(":")[-1] if ":" in device else "0"
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
            logger.info("Starting SGLang on GPU %s: %s", gpu_idx, " ".join(cmd))
        else:
            logger.info("Starting SGLang (TP=%d): %s", tp, " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        self._managed = True

        # Wait for SGLang to become healthy
        start = time.monotonic()
        timeout = 600
        last_log = 0
        try:
            while time.monotonic() - start < timeout:
                if self._process.poll() is not None:
                    output = self._process.stdout.read() if self._process.stdout else ""
                    logger.error("SGLang process exited with code %d", self._process.returncode)
                    logger.error("SGLang output (last 500 chars): %s", output[-500:])
                    self._kill_process()
                    raise RuntimeError(f"SGLang failed to start: {output[-200:]}")

                if await self._health_check():
                    elapsed = int(time.monotonic() - start)
                    self._model = True
                    logger.info("SGLang ready in %ds at %s", elapsed, self._base_url)
                    return

                elapsed_now = int(time.monotonic() - start)
                if elapsed_now - last_log >= 30:
                    last_log = elapsed_now
                    logger.info("SGLang still starting... (%ds elapsed)", elapsed_now)

                await asyncio.sleep(5)

            logger.error("SGLang did not become healthy within %ds", timeout)
            self._kill_process()
            raise RuntimeError(f"SGLang did not become healthy within {timeout}s")
        except Exception:
            self._kill_process()
            raise

    def unload(self) -> None:
        """Kill SGLang subprocess and release GPU memory."""
        if self._managed and (self._process is not None or self._adopted_pid is not None):
            logger.info("Unloading SGLang model: killing process (port %s)", self._port)
            self._kill_process()
            logger.info("SGLang process killed, GPU memory released")
        else:
            logger.info("Disconnecting from external SGLang at %s", self._base_url)
        self._model = None

    def _kill_process(self) -> None:
        import signal

        if self._process is not None:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=5)
            except Exception as e:
                logger.warning("Error killing SGLang subprocess: %s", e)
            finally:
                self._process = None
            return

        if self._adopted_pid:
            try:
                pgid = os.getpgid(self._adopted_pid)
                os.killpg(pgid, signal.SIGTERM)
                logger.info("Sent SIGTERM to adopted SGLang (pid=%d)", self._adopted_pid)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning("Error killing adopted SGLang (pid=%d): %s", self._adopted_pid, e)
            finally:
                self._adopted_pid = None

    @property
    def pid(self) -> int | None:
        if self._process is not None:
            return self._process.pid
        return self._adopted_pid

    async def _health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    async def infer(self, params: dict[str, Any]) -> InferenceResult:
        """Forward chat completion request to SGLang."""
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions", json=params
        )
        return InferenceResult(data=resp.content, content_type="application/json")

    async def infer_stream(self, params: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream SSE chunks from SGLang."""
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            json={**params, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line.encode() + b"\n"
