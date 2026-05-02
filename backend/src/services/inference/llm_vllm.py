"""VLLMAdapter — manages vLLM as a subprocess with full lifecycle control."""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import subprocess
import sys
import time
import urllib.parse
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from src.services.inference.base import (
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    StreamEvent,
    TextRequest,
    UsageMeter,
)
from src.utils.constants import ALLOWED_LLM_HOSTS

logger = logging.getLogger(__name__)


class VLLMAdapter(InferenceAdapter):
    """Adapter that spawns vLLM as a subprocess and manages its lifecycle.

    load()         → start vLLM subprocess → wait for health check
    unload()       → kill subprocess → free GPU memory
    infer(req)     → HTTP POST /v1/chat/completions, return InferenceResult
    infer_stream() → SSE stream → yields StreamEvent("delta"|"done")
    """

    modality = MediaModality.TEXT
    estimated_vram_mb = 0  # Determined at runtime

    def __init__(
        self,
        paths: dict[str, str],
        device: str = "cuda",
        vllm_base_url: str | None = None,
        vllm_port: int | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        quantization: str | None = None,
        dtype: str | None = None,
        max_num_seqs: int | None = None,
        enable_prefix_caching: bool | None = None,
        adopt_pid: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(paths=paths, device=device)
        # Single-component model: 'main' is the HF model dir under LOCAL_MODELS_PATH
        self.model_path = Path(paths.get("main", ""))
        self._port = vllm_port or (int(vllm_base_url.split(":")[-1]) if vllm_base_url else 0)
        self._tp = tensor_parallel_size
        self._max_model_len = max_model_len
        self._gpu_mem_util = gpu_memory_utilization
        self._quantization = quantization
        self._max_num_seqs = max_num_seqs
        self._dtype = dtype
        # If True, vLLM is launched with --enable-prefix-caching.
        # Per-model override; reads from models.yaml `params` block.
        self._enable_prefix_caching = enable_prefix_caching
        # Port resolved lazily in load() if not set
        if self._port:
            self._base_url = f"http://localhost:{self._port}"
        else:
            self._base_url = None  # Will be set in load()
        self.base_url = self._base_url
        self._process: subprocess.Popen | None = None
        self._adopt_pid = adopt_pid  # PID of an orphan process to adopt
        self._adopted_pid: int | None = None  # Set in load() when adopting
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
        # Use gptq_marlin for faster inference (vLLM recommended over plain gptq)
        if quantization == "gptq":
            quantization = "gptq_marlin"

        # 4. Auto-detect dtype — let vLLM choose (bfloat16 is safer for mixed-dtype GPTQ models)
        dtype = None

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

        # 9. Calculate max_model_len based on available KV cache memory
        # KV cache per token varies by model, use ~128KB/token as estimate
        kv_bytes_per_token = 131072  # 128KB, typical for Qwen3.5/Gemma4 MoE
        kv_cache_bytes = kv_buffer_gb * 1024**3
        estimated_max = int(kv_cache_bytes / kv_bytes_per_token)
        # Read model's native max from config
        max_position = model_config.get("max_position_embeddings") or \
                       model_config.get("text_config", {}).get("max_position_embeddings", 262144)
        # Use the smaller of estimated capacity and model native max, round down to 1024
        max_model_len = min(estimated_max, max_position)
        max_model_len = max(2048, (max_model_len // 1024) * 1024)  # at least 2048

        # 10. Calculate max_num_seqs (conservative to avoid sampler warmup OOM)
        if kv_buffer_gb < 3.0:
            max_num_seqs = 16
        elif kv_buffer_gb < 6.0:
            max_num_seqs = 32
        else:
            max_num_seqs = 64

        # 11. Find free port
        port = self._port
        if not port:
            import socket
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        # 12. Detect multimodal (vision-language) models
        archs = model_config.get("architectures") or []
        is_multimodal = any(
            "VL" in a or "Vision" in a or "Multimodal" in a or "Omni" in a
            for a in archs
        ) or model_config.get("vision_config") is not None

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
            "is_multimodal": is_multimodal,
        }

    async def load(self, device: str | None = None) -> None:
        """Start vLLM subprocess or connect to existing instance."""
        # First check if vLLM is already running on this port
        if self._base_url and await self._health_check():
            self._model = True
            if self._adopt_pid:
                # Adopt orphan process — we manage its lifecycle
                self._managed = True
                self._adopted_pid = self._adopt_pid
                logger.info("Adopted orphan vLLM (pid=%d) at %s", self._adopt_pid, self._base_url)
            else:
                self._managed = False  # External instance, don't kill it
                logger.info("Connected to existing vLLM at %s", self._base_url)
            return

        # Auto-configure parameters
        auto = self._auto_configure(device)
        port = self._port or auto["port"]
        tp = self._tp or auto["tp"]
        max_model_len = self._max_model_len or auto["max_model_len"]
        self.max_model_len = max_model_len  # expose for clamp logic
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
        if self._enable_prefix_caching:
            # Repeated system prompts / few-shot examples reuse cached KV
            # blocks instead of re-prefilling. Memory cost is tiny metadata;
            # benefit is large when callers send the same prefix often.
            cmd += ["--enable-prefix-caching"]
        if auto.get("is_multimodal"):
            # vLLM >=0.6 parses this value with json.loads — must be JSON, not key=val.
            # Allow up to 4 images per prompt by default.
            cmd += ["--limit-mm-per-prompt", '{"image":4}']
            self.is_multimodal = True
            logger.info('Detected multimodal model — enabling --limit-mm-per-prompt {"image":4}')
        else:
            self.is_multimodal = False

        # Set cache directories to persistent storage (avoid re-compilation)
        env = dict(os.environ)
        from src.config import get_settings
        _cache_root = str(Path(get_settings().LOCAL_MODELS_PATH) / ".cache")
        env["TORCH_HOME"] = str(Path(_cache_root) / "torch")
        env["XDG_CACHE_HOME"] = _cache_root

        # Set CUDA_VISIBLE_DEVICES for single-GPU mode
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
            start_new_session=True,  # Create process group for clean kill
        )
        self._managed = True

        # Wait for vLLM to become healthy (up to 10 minutes for first-time CUDA kernel compilation)
        start = time.monotonic()
        timeout = 600
        last_log = 0
        try:
            while time.monotonic() - start < timeout:
                if self._process.poll() is not None:
                    # Process exited
                    output = self._process.stdout.read() if self._process.stdout else ""
                    logger.error("vLLM process exited with code %d", self._process.returncode)
                    logger.error("vLLM output (last 500 chars): %s", output[-500:])
                    self._kill_process()
                    raise RuntimeError(f"vLLM failed to start: {output[-200:]}")

                if await self._health_check():
                    elapsed = int(time.monotonic() - start)
                    self._model = True
                    logger.info("vLLM ready in %ds at %s", elapsed, self._base_url)
                    return

                # Log progress every 30 seconds
                elapsed_now = int(time.monotonic() - start)
                if elapsed_now - last_log >= 30:
                    last_log = elapsed_now
                    logger.info("vLLM still starting... (%ds elapsed, timeout %ds)", elapsed_now, timeout)

                await asyncio.sleep(5)

            # Timeout
            logger.error("vLLM did not become healthy within %ds", timeout)
            self._kill_process()
            raise RuntimeError(f"vLLM did not become healthy within {timeout}s")
        except Exception:
            # Ensure cleanup on ANY failure
            self._kill_process()
            raise

    def unload(self) -> None:
        """Kill vLLM subprocess and release GPU memory."""
        if self._managed and (self._process is not None or self._adopted_pid is not None):
            logger.info("Unloading vLLM model: killing process (port %s)", self._port)
            self._kill_process()
            logger.info("vLLM process killed, GPU memory released")
        else:
            logger.info("Disconnecting from external vLLM at %s", self._base_url)
        self._model = None

    def _kill_process(self) -> None:
        import signal

        # Kill subprocess we spawned
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
                logger.warning("Error killing vLLM subprocess: %s", e)
            finally:
                self._process = None
            return

        # Kill adopted orphan process
        adopted = getattr(self, "_adopted_pid", None)
        if adopted:
            try:
                pgid = os.getpgid(adopted)
                os.killpg(pgid, signal.SIGTERM)
                logger.info("Sent SIGTERM to adopted vLLM process group (pid=%d)", adopted)
            except ProcessLookupError:
                pass  # Already gone
            except Exception as e:
                logger.warning("Error killing adopted vLLM (pid=%d): %s", adopted, e)
            finally:
                self._adopted_pid = None

    @property
    def pid(self) -> int | None:
        """Return the PID of the managed vLLM process, if any."""
        if self._process is not None:
            return self._process.pid
        return self._adopted_pid

    async def _health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _validate_base_url(self) -> None:
        """vLLM only on localhost (defense-in-depth — admin-controlled config)."""
        parsed = urllib.parse.urlparse(self._base_url or "")
        if parsed.hostname and parsed.hostname not in ALLOWED_LLM_HOSTS:
            raise ValueError(f"vLLM base_url 只允许 localhost，收到: {parsed.hostname}")

    def _clamp_max_tokens(self, requested: int) -> int:
        """Per-model max_model_len enforcement (replaces TextRequest schema ceiling).

        Outside-voice #7a: 200k-context models must not be rejected at schema layer.
        """
        model_max = getattr(self, "max_model_len", None) or 4096
        safe_max = max(model_max - 512, model_max // 2)
        return min(requested, safe_max)

    def _build_payload(self, req: TextRequest) -> dict[str, Any]:
        return {
            "model": req.model,
            "messages": [m.model_dump(mode="json") for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": self._clamp_max_tokens(req.max_tokens),
            # Always pass explicit value — Qwen3's chat template defaults to
            # thinking=True; omitting the flag still produces reasoning traces.
            "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
            **req.extra,
        }

    def _build_headers(self, req: TextRequest) -> dict[str, str]:
        if req.api_key:
            return {"Authorization": f"Bearer {req.api_key}"}
        return {}

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        """Non-streaming chat completion. Wraps vLLM /v1/chat/completions."""
        if not isinstance(req, TextRequest):
            raise TypeError(f"VLLMAdapter expects TextRequest, got {type(req).__name__}")
        self._validate_base_url()

        t0 = time.monotonic()
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json=self._build_payload(req),
            headers=self._build_headers(req),
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                detail = resp.text[:300]
            raise RuntimeError(f"vLLM API error ({resp.status_code}): {detail}")

        body = resp.json()
        usage_dict = body.get("usage") or {}
        usage = UsageMeter(
            input_tokens=usage_dict.get("prompt_tokens"),
            output_tokens=usage_dict.get("completion_tokens"),
            latency_ms=latency_ms,
        )
        return InferenceResult(
            media_type="application/json",
            data=resp.content,
            metadata={"raw": body},
            usage=usage,
        )

    async def infer_stream(self, req: InferenceRequest) -> AsyncIterator[StreamEvent]:
        """SSE stream → yields StreamEvent('delta', {chunk}) / ('done', {usage})."""
        if not isinstance(req, TextRequest):
            raise TypeError(f"VLLMAdapter expects TextRequest, got {type(req).__name__}")
        self._validate_base_url()

        payload = self._build_payload(req)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})

        last_usage: dict[str, Any] | None = None
        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            headers=self._build_headers(req),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield StreamEvent(
                    type="error",
                    payload={"status_code": resp.status_code, "body": body[:300].decode("utf-8", errors="replace")},
                )
                return
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    chunk = _json.loads(payload_text)
                except _json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    last_usage = chunk["usage"]
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta") if choices else None
                if delta:
                    yield StreamEvent(type="delta", payload=delta)
        yield StreamEvent(type="done", payload={"usage": last_usage or {}})
