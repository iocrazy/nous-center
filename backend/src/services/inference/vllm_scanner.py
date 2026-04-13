"""Scan running vLLM processes by reading OS process list."""
from __future__ import annotations

import logging
import re
import subprocess

import httpx

logger = logging.getLogger(__name__)


def scan_running_vllm() -> list[dict]:
    """Return ALL running vLLM instances with health status.

    Each entry: {"model_path": str, "port": int, "pid": int, "healthy": bool}
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
            if "vllm.entrypoints" not in line:
                continue

            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            cmdline = parts[1]

            model_match = re.search(r"--model\s+(\S+)", cmdline)
            port_match = re.search(r"--port\s+(\d+)", cmdline)
            if model_match and port_match:
                candidates.append({
                    "model_path": model_match.group(1),
                    "port": int(port_match.group(1)),
                    "pid": pid,
                })
    except Exception as e:
        logger.warning("Failed to scan vLLM processes: %s", e)
        return []

    # Health-check each candidate
    results: list[dict] = []
    for c in candidates:
        healthy = False
        try:
            resp = httpx.get(f"http://localhost:{c['port']}/v1/models", timeout=3)
            healthy = resp.status_code == 200
        except Exception:
            pass

        c["healthy"] = healthy
        results.append(c)
        logger.info(
            "Found vLLM: pid=%d, model=%s, port=%d, healthy=%s",
            c["pid"], c["model_path"], c["port"], healthy,
        )

    return results
