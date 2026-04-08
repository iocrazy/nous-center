from __future__ import annotations
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class GPUAllocator:
    def __init__(self, poll_fn: Callable[[], list[dict]] | None = None):
        if poll_fn is None:
            from src.services.gpu_monitor import poll_gpu_stats
            poll_fn = poll_gpu_stats
        self._poll = poll_fn

    def get_best_gpu(self, required_vram_mb: float) -> int:
        stats = self._poll()
        if not stats:
            return -1
        candidates = [(s["index"], s["free_mb"]) for s in stats if s["free_mb"] >= required_vram_mb]
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_free_mb(self, gpu_index: int) -> int:
        stats = self._poll()
        for s in stats:
            if s["index"] == gpu_index:
                return s["free_mb"]
        return 0
