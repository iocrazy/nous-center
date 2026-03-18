import logging
import os
import socket
import subprocess
import time

from src.workers.llm_engines.base import LLMEngine
from src.workers.llm_engines.registry import register_engine

logger = logging.getLogger(__name__)


@register_engine
class VLLMEngine(LLMEngine):
    """Manages a vLLM subprocess serving an OpenAI-compatible API."""

    ENGINE_NAME = "vllm"

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(model_path, device, **kwargs)
        self._port: int = kwargs.get("port", 0)  # 0 = auto-assign
        self._tensor_parallel: int = kwargs.get("tensor_parallel_size", 1)
        self._base_url: str | None = None

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def load(self) -> None:
        if self._port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self._port = s.getsockname()[1]

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--port", str(self._port),
            "--host", "127.0.0.1",
        ]
        if self._tensor_parallel > 1:
            cmd += ["--tensor-parallel-size", str(self._tensor_parallel)]

        env = dict(os.environ)

        # Parse device for GPU assignment
        if self.device.startswith("cuda:"):
            gpu_id = self.device.split(":")[1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        elif "," in str(self._config.get("gpu", "")):
            # Multi-GPU
            gpu_ids = str(self._config["gpu"]).strip("[]").replace(" ", "")
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        logger.info("Starting vLLM: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._base_url = f"http://127.0.0.1:{self._port}"

        # Wait for server to be ready (poll health endpoint)
        import httpx

        for _ in range(120):  # 2 min timeout
            try:
                r = httpx.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info("vLLM server ready at %s", self._base_url)
                    return
            except Exception:
                pass
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                self._process = None
                raise RuntimeError(f"vLLM process exited: {stderr[:500]}")
            time.sleep(1)

        # Timeout — kill the process
        self._process.terminate()
        self._process = None
        raise RuntimeError("vLLM server did not start within 2 minutes")

    async def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        from src.services.llm_service import call_llm

        if not self._base_url:
            raise RuntimeError("vLLM engine is not loaded")

        return await call_llm(
            prompt=prompt,
            base_url=self._base_url,
            system=system,
            **kwargs,
        )

    def unload(self) -> None:
        self._base_url = None
        super().unload()

    @property
    def engine_name(self) -> str:
        return "vllm"
