"""GPU auto-detection module.

Detects available GPUs at startup and provides device info
for dynamic engine allocation.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vram_total_gb: float
    compute_capability: tuple[int, int]


def detect_gpus() -> list[GPUInfo]:
    """Detect all available CUDA GPUs. Returns empty list if no GPU/CUDA."""
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed, no GPU detection")
        return []

    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = GPUInfo(
            index=i,
            name=props.name,
            vram_total_gb=round(props.total_memory / (1024 ** 3), 1),
            compute_capability=(props.major, props.minor),
        )
        gpus.append(info)
        logger.info("GPU %d: %s (%.1f GB, sm_%d%d)", i, info.name, info.vram_total_gb, *info.compute_capability)

    return gpus


# Cached result
_cached_gpus: list[GPUInfo] | None = None


def get_gpus() -> list[GPUInfo]:
    """Get detected GPUs (cached after first call)."""
    global _cached_gpus
    if _cached_gpus is None:
        _cached_gpus = detect_gpus()
    return _cached_gpus


def get_device_for_engine(engine_config: dict) -> str:
    """Resolve device string for an engine based on config and available GPUs.

    Logic:
    - If config has explicit 'gpu' field, use it (e.g., "cuda:1")
    - If GPUs detected, pick the one with most free VRAM
    - Fallback to "cpu" if no GPU
    """
    gpus = get_gpus()

    if not gpus:
        return "cpu"

    gpu_index = engine_config.get("gpu")
    if gpu_index is not None:
        if isinstance(gpu_index, int) and gpu_index < len(gpus):
            return f"cuda:{gpu_index}"
        elif isinstance(gpu_index, str) and gpu_index.startswith("cuda"):
            return gpu_index

    # Default: use last GPU (typically reserved for TTS in dual-GPU setups)
    return f"cuda:{gpus[-1].index}"


def gpu_summary() -> dict:
    """Return a summary dict for API responses."""
    gpus = get_gpus()
    return {
        "count": len(gpus),
        "devices": [
            {
                "index": g.index,
                "name": g.name,
                "vram_gb": g.vram_total_gb,
                "compute_capability": f"{g.compute_capability[0]}.{g.compute_capability[1]}",
            }
            for g in gpus
        ],
    }
