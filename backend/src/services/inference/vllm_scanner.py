"""Scan running vLLM processes by reading OS process list."""
from __future__ import annotations

import logging
import re
import subprocess

import httpx

logger = logging.getLogger(__name__)


def scan_running_vllm() -> list[dict]:
    """Return ALL running vLLM instances with health status.

    Each entry: {"model_path": str, "port": int, "pid": int,
                 "tensor_parallel_size": int, "healthy": bool}

    ``tensor_parallel_size > 1`` = 这个进程横跨多张卡(cmdline 里带
    ``--tensor-parallel-size``),不是「一个模型一张卡」。
    """
    candidates: list[dict] = []
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True, text=True, timeout=5,
        )
        if output.returncode != 0:
            return []

        for line in output.stdout.strip().split("\n"):
            line = line.strip()
            if "vllm.entrypoints" not in line and "sglang.launch_server" not in line:
                continue

            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            cmdline = parts[1]

            # Match both vLLM (--model) and SGLang (--model-path)
            model_match = re.search(r"--model(?:-path)?\s+(\S+)", cmdline)
            port_match = re.search(r"--port\s+(\d+)", cmdline)
            # 多卡(张量并行)进程识别:cmdline 带 --tensor-parallel-size N(N>1)
            # 就是一个占 N 张卡的进程。orphan adopt / 看护要知道它不只占一张卡,
            # 否则「按单卡口径」清理会漏掉副卡上的残留。SGLang 的是 --tp-size。
            tp_match = re.search(r"--(?:tensor-parallel-size|tp-size)[= ]+(\d+)", cmdline)
            if model_match and port_match:
                candidates.append({
                    "model_path": model_match.group(1),
                    "port": int(port_match.group(1)),
                    "pid": pid,
                    "tensor_parallel_size": int(tp_match.group(1)) if tp_match else 1,
                })
    except Exception as e:
        logger.warning("Failed to scan vLLM processes: %s", e)
        return []

    # Health-check each candidate
    results: list[dict] = []
    for c in candidates:
        healthy = False
        try:
            # trust_env=False:localhost 探活别经本机代理(round3 #2;否则代理拦截 →
            # 每个在跑的实例都被误判 unhealthy,orphan adopt/reconnect 永不匹配)。
            resp = httpx.get(
                f"http://localhost:{c['port']}/v1/models", timeout=3, trust_env=False
            )
            healthy = resp.status_code == 200
        except Exception:
            pass

        c["healthy"] = healthy
        results.append(c)
        logger.info(
            "Found vLLM: pid=%d, model=%s, port=%d, tp=%d, healthy=%s",
            c["pid"], c["model_path"], c["port"],
            c.get("tensor_parallel_size", 1), healthy,
        )

    return results
