"""GPU memory monitor — polls nvidia-smi and enforces memory limits."""

import asyncio
import logging
import subprocess
import threading
import time

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


async def check_and_evict(model_manager, reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """检查 GPU 显存，低于阈值时让 model_manager 驱逐该 GPU 上的 LRU 模型。

    model_manager: services.model_manager.ModelManager 实例（evict_lru 内部已处理
    resident / referenced 跳过 + last_used 排序 + force unload）。
    """
    stats = poll_gpu_stats()
    for gpu in stats:
        free_gb = gpu["free_mb"] / 1024
        if free_gb < reserved_gb:
            logger.debug(
                "GPU %d low memory: %.1fGB free (threshold: %.1fGB). Evicting LRU...",
                gpu["index"], free_gb, reserved_gb,
            )
            evicted = await model_manager.evict_lru(gpu_index=gpu["index"])
            if evicted:
                logger.info("Auto-evicted model %s from GPU %d", evicted, gpu["index"])


async def memory_guard_loop(model_manager, reserved_gb: float = DEFAULT_RESERVED_GB) -> None:
    """后台 loop：每 POLL_INTERVAL_SECONDS 检查一次 GPU 显存。"""
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            await check_and_evict(model_manager, reserved_gb)
        except Exception as e:
            logger.warning("GPU memory guard failed: %s", e)
