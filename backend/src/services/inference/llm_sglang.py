"""SGLangAdapter — manages SGLang as a subprocess with full lifecycle control.

Drop-in replacement for VLLMAdapter. SGLang serves an OpenAI-compatible API
and is 3-5x faster than vLLM on Qwen3.5 MoE models.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
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


def _proc_cmdline_is_sglang(pid: int) -> bool:
    """Recycled-PID guard for adopted SGLang orphans (see safe_signal)."""
    from src.services.safe_signal import _proc_cmdline_contains
    return _proc_cmdline_contains(pid, "sglang")


class SGLangAdapter(InferenceAdapter):
    """Adapter that spawns SGLang as a subprocess and manages its lifecycle."""

    modality = MediaModality.TEXT
    estimated_vram_mb = 0

    def __init__(
        self,
        paths: dict[str, str],
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
        super().__init__(paths=paths, device=device)
        self.model_path = Path(paths.get("main", ""))
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
        # trust_env=False:只连本地 SGLang 子进程;proxy=None 不够(env proxy 在请求期
        # 解析),trust_env=False 才彻底绕开本机代理(round3 #2,与 vLLM adapter 取齐)。
        self._client = httpx.AsyncClient(
            timeout=120, limits=httpx.Limits(max_connections=10), trust_env=False
        )
        self._managed = True
        # round3 #1:后台抽干 stdout 防 PIPE 填满死锁(同 vLLM adapter)。
        self._stdout_tail: deque[str] = deque(maxlen=200)
        self._drain_thread: threading.Thread | None = None

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
        if tp <= 1 and model_size_gb > gpu_free_gb * 0.6:
            total_free = sum(g["free_mb"] for g in gpu_stats) / 1024
            if total_free > model_size_gb * 1.2:
                tp = len(gpu_stats)

        if tp > 1:
            per_gpu_model = model_size_gb / tp
            kv_buffer_gb = 4.0
            # SGLang mem-fraction-static = fraction of FREE memory (after model) for KV cache
            free_after_model = gpu_total_gb - per_gpu_model - 1.0  # 1GB CUDA overhead
            utilization = min(0.80, kv_buffer_gb / free_after_model) if free_after_model > 0 else 0.5
        else:
            kv_buffer_gb = min(4.0, gpu_free_gb - model_size_gb - 2.0)  # 2GB reserved
            if kv_buffer_gb < 1.0:
                kv_buffer_gb = 1.0
            free_after_model = gpu_free_gb - model_size_gb - 1.0
            utilization = min(0.75, kv_buffer_gb / free_after_model) if free_after_model > 0 else 0.4

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
            # 回填 max_model_len(同 vLLM):快速返回路径不设 → _clamp_max_tokens 退 4096
            # → 重连后长输出被砍。优先 yaml 配的,否则从 /v1/models 读。
            self.max_model_len = (
                self._max_model_len or await self._fetch_remote_max_model_len() or 4096)
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
        self.max_model_len = max_model_len
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

        # Build SGLang command — let SGLang auto-manage memory allocation
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--context-length", str(max_model_len),
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
        # 启动后台抽干线程(daemon),防 stdout PIPE 填满死锁。
        self._stdout_tail.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_stdout, name=f"sglang-stdout-{self._port}", daemon=True
        )
        self._drain_thread.start()

        # Wait for SGLang to become healthy
        start = time.monotonic()
        timeout = 600
        last_log = 0
        try:
            while time.monotonic() - start < timeout:
                if self._process.poll() is not None:
                    # 等抽干线程收尾后取尾部日志(stdout 已被 drain 线程消费)。
                    if self._drain_thread is not None:
                        self._drain_thread.join(timeout=1.0)
                    output = "".join(self._stdout_tail)
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

    def _drain_stdout(self) -> None:
        """后台线程:持续读子进程 stdout 进有界 deque,防 PIPE 填满阻塞(同 vLLM adapter)。"""
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._stdout_tail.append(line)
        except Exception:  # noqa: BLE001
            pass

    def _kill_process(self) -> None:
        import signal
        from src.services.safe_signal import safe_killpg

        # safe_killpg refuses pgid<=1 (broadcast guard: killpg(1)/killpg(0) would
        # signal every process the user can reach — took down mihomo/sshd/systemd).
        if self._process is not None:
            try:
                if safe_killpg(self._process.pid, signal.SIGTERM):
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        safe_killpg(self._process.pid, signal.SIGKILL)
                        self._process.wait(timeout=5)
            except Exception as e:
                logger.warning("Error killing SGLang subprocess: %s", e)
            finally:
                self._process = None
            return

        if self._adopted_pid:
            try:
                if safe_killpg(
                    self._adopted_pid, signal.SIGTERM,
                    verify=lambda p: _proc_cmdline_is_sglang(p),
                ):
                    logger.info("Sent SIGTERM to adopted SGLang (pid=%d)", self._adopted_pid)
                else:
                    logger.warning(
                        "Refused to kill adopted SGLang PID %d (recycled or broadcast guard).",
                        self._adopted_pid,
                    )
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

    async def _fetch_remote_max_model_len(self) -> int | None:
        """从运行中 SGLang 的 /v1/models 读 max_model_len(重连/adopt 时 yaml 没配的兜底)。"""
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models", timeout=3)
            if resp.status_code != 200:
                return None
            for m in (resp.json().get("data") or []):
                v = m.get("max_model_len")
                if isinstance(v, int) and v > 0:
                    return v
        except Exception:  # noqa: BLE001
            return None
        return None

    def _validate_base_url(self) -> None:
        parsed = urllib.parse.urlparse(self._base_url or "")
        if parsed.hostname and parsed.hostname not in ALLOWED_LLM_HOSTS:
            raise ValueError(f"SGLang base_url 只允许 localhost，收到: {parsed.hostname}")

    def _clamp_max_tokens(self, requested: int) -> int:
        model_max = getattr(self, "max_model_len", None) or 4096
        safe_max = max(model_max - 512, model_max // 2)
        return min(requested, safe_max)

    def _build_payload(self, req: TextRequest) -> dict[str, Any]:
        return {
            "model": req.model,
            "messages": [m.model_dump(mode="json") for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": self._clamp_max_tokens(req.max_tokens),
            "chat_template_kwargs": {"enable_thinking": req.enable_thinking},
            **req.extra,
        }

    def _build_headers(self, req: TextRequest) -> dict[str, str]:
        if req.api_key:
            return {"Authorization": f"Bearer {req.api_key}"}
        return {}

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        if not isinstance(req, TextRequest):
            raise TypeError(f"SGLangAdapter expects TextRequest, got {type(req).__name__}")
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
            raise RuntimeError(f"SGLang API error ({resp.status_code}): {detail}")

        body = resp.json()
        # round9:200-但-body-级-error(OpenAI 错误体)同 vLLM —— 只判 status 会静默吐空。
        if isinstance(body, dict) and (body.get("object") == "error" or body.get("error")):
            err = body.get("message") or body.get("error") or "unknown error"
            if isinstance(err, dict):
                err = err.get("message") or str(err)
            raise RuntimeError(f"SGLang API error (200 body): {err}")
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
        if not isinstance(req, TextRequest):
            raise TypeError(f"SGLangAdapter expects TextRequest, got {type(req).__name__}")
        self._validate_base_url()

        payload = self._build_payload(req)
        payload["stream"] = True
        # round9:**req.extra 可塞 stream_options 盖掉这里,setdefault 不纠正 → usage 丢、
        # 计费拿空。强制合并 include_usage=True,保留调用方其它键(与 vLLM 一致)。
        _so = dict(payload.get("stream_options") or {})
        _so["include_usage"] = True
        payload["stream_options"] = _so

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
