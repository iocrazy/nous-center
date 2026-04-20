"""GPU auto-detection module.

Detects available GPUs at startup and provides device info
for dynamic engine allocation.
"""

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Process names that mean "this GPU is driving a display server" — compute
# workloads allocated to it fight Xorg/Wayland for VRAM and can hard-crash
# the session on NVIDIA drivers under pressure.
_DISPLAY_PROCESS_NAMES = (
    "Xorg",
    "X",
    "gnome-shell",
    "gdm-x-session",
    "Hyprland",
    "sway",
    "kwin_x11",
    "kwin_wayland",
    "plasmashell",
)


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


def get_display_gpu_indices() -> set[int]:
    """Detect which GPU indices are driving an active display server.

    Uses `nvidia-smi --query-compute-apps` to list processes on each GPU,
    flags any with a known compositor/X server name. Returns an empty set
    on failure (tool missing, parse error, etc.) — callers must not depend
    on this being populated.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return set()

        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if uuid_result.returncode != 0:
            return set()

        uuid_to_idx: dict[str, int] = {}
        for line in uuid_result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    uuid_to_idx[parts[1]] = int(parts[0])
                except ValueError:
                    continue

        display: set[int] = set()
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            uuid, proc_name = parts[0], parts[1]
            if any(name in proc_name for name in _DISPLAY_PROCESS_NAMES):
                idx = uuid_to_idx.get(uuid)
                if idx is not None:
                    display.add(idx)
        return display
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return set()


def get_device_for_engine(engine_config: dict) -> str:
    """Resolve device string for an engine based on config and available GPUs.

    Logic:
    - If config has explicit 'gpu' field, honor it (and warn if that GPU
      is driving a display server — TTS/LLM on the X GPU has crashed sessions)
    - Otherwise pick the first non-display GPU
    - Fallback to first GPU if all appear to be driving displays
    - "cpu" if no GPU detected
    """
    gpus = get_gpus()

    if not gpus:
        return "cpu"

    display_indices = get_display_gpu_indices()

    gpu_index = engine_config.get("gpu")
    if gpu_index is not None:
        if isinstance(gpu_index, int) and gpu_index < len(gpus):
            if gpu_index in display_indices:
                logger.warning(
                    "Engine pinned to cuda:%d but that GPU is driving a display "
                    "server — compute load here can hang the desktop.", gpu_index,
                )
            return f"cuda:{gpu_index}"
        elif isinstance(gpu_index, str) and gpu_index.startswith("cuda"):
            return gpu_index

    # Auto: prefer any non-display GPU, in index order.
    for g in gpus:
        if g.index not in display_indices:
            return f"cuda:{g.index}"

    # All GPUs appear to drive a display (or detection failed). Fall back to
    # gpus[0] — matches single-GPU workstations and single-compute-card setups.
    logger.warning(
        "No non-display GPU found (display GPUs: %s); falling back to cuda:%d",
        sorted(display_indices) or "none", gpus[0].index,
    )
    return f"cuda:{gpus[0].index}"


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
