"""GPU memory monitor — polls nvidia-smi and enforces memory limits."""

import asyncio
import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default: leave 4GB free per GPU
DEFAULT_RESERVED_GB = 4.0
POLL_INTERVAL_SECONDS = 5

_gpu_stats: list[dict] = []  # cached GPU stats
_last_poll: float = 0
_stats_lock = threading.Lock()  # protects _gpu_stats and _last_poll


def poll_gpu_stats() -> list[dict]:
    """Query nvidia-smi for current GPU memory usage."""
    global _gpu_stats, _last_poll

    with _stats_lock:
        now = time.time()
        if now - _last_poll < 2:  # Cache for 2 seconds
            return list(_gpu_stats)

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return list(_gpu_stats)

            stats = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                stats.append(
                    {
                        "index": int(parts[0]),
                        "used_mb": int(parts[1]),
                        "total_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                        "utilization_pct": int(parts[4]),
                        "temperature": int(parts[5]),
                    }
                )
            _gpu_stats = stats
            _last_poll = now
        except Exception as e:
            logger.warning("nvidia-smi poll failed: %s", e)

        return list(_gpu_stats)


def get_gpu_free_mb(gpu_index: int) -> int:
    """Get free memory in MB for a specific GPU."""
    stats = poll_gpu_stats()
    for s in stats:
        if s["index"] == gpu_index:
            return s["free_mb"]
    return 0


def get_gpu_stats() -> list[dict]:
    """Get cached GPU stats."""
    return poll_gpu_stats()


async def check_and_evict(reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """Check GPU memory and evict LRU models if free memory is below threshold."""
    from src.services import model_scheduler
    from src.config import load_model_configs

    stats = poll_gpu_stats()
    configs = load_model_configs()

    for gpu in stats:
        free_gb = gpu["free_mb"] / 1024
        if free_gb < reserved_gb:
            logger.warning(
                "GPU %d low memory: %.1fGB free (threshold: %.1fGB). Evicting LRU model...",
                gpu["index"],
                free_gb,
                reserved_gb,
            )

            # Find loaded models on this GPU, sorted by last_used (oldest first)
            candidates = []
            async with model_scheduler._lock:
                for model_key in list(model_scheduler._loaded_models):
                    cfg = configs.get(model_key, {})
                    model_gpu = cfg.get("gpu", -1)
                    if isinstance(model_gpu, list):
                        on_this_gpu = gpu["index"] in model_gpu
                    else:
                        on_this_gpu = model_gpu == gpu["index"]

                    if not on_this_gpu:
                        continue
                    if cfg.get("resident", False):
                        continue  # Don't evict resident models
                    if model_scheduler._references.get(model_key):
                        continue  # Don't evict referenced models

                    last_used = model_scheduler._last_used.get(model_key, 0)
                    candidates.append((model_key, last_used))

            # Sort by last_used ascending (oldest first)
            candidates.sort(key=lambda x: x[1])

            # Evict oldest
            for model_key, _ in candidates:
                logger.info(
                    "Auto-evicting model %s from GPU %d", model_key, gpu["index"]
                )
                await model_scheduler.unload_model(model_key, force=True)
                break  # Evict one at a time, re-check next cycle


async def memory_guard_loop(reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """Background loop: check GPU memory every POLL_INTERVAL_SECONDS."""
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            await check_and_evict(reserved_gb)
        except Exception as e:
            logger.warning("GPU memory guard failed: %s", e)
