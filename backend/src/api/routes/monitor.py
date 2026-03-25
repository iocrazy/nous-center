"""Real-time system resource monitoring endpoint."""

from __future__ import annotations

import subprocess
import time

import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])


def _gpu_stats_nvidia_smi() -> list[dict] | None:
    """Query nvidia-smi for GPU stats. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            idx = int(parts[0])
            mem_used = int(parts[2])
            mem_total = int(parts[3])
            gpus.append(
                {
                    "index": idx,
                    "name": parts[1],
                    "utilization_gpu": int(parts[4]),
                    "utilization_memory": round(mem_used / mem_total * 100, 1)
                    if mem_total > 0
                    else 0,
                    "temperature": int(parts[5]),
                    "fan_speed": int(parts[6]) if parts[6] != "[N/A]" else 0,
                    "power_draw_w": float(parts[7]) if parts[7] != "[N/A]" else 0,
                    "power_limit_w": float(parts[8]) if parts[8] != "[N/A]" else 0,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_free_mb": mem_total - mem_used,
                    "processes": [],
                }
            )
        return gpus
    except Exception:
        return None


def _gpu_processes() -> dict[int, list[dict]]:
    """Get per-GPU process memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}

        # Map GPU UUID -> index
        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        uuid_to_idx: dict[str, int] = {}
        if uuid_result.returncode == 0:
            for line in uuid_result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    uuid_to_idx[parts[1]] = int(parts[0])

        procs: dict[int, list[dict]] = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid = parts[0]
            gpu_idx = uuid_to_idx.get(gpu_uuid, -1)
            if gpu_idx < 0:
                continue
            procs.setdefault(gpu_idx, []).append(
                {
                    "pid": int(parts[1]),
                    "used_gpu_memory_mb": int(parts[2]),
                }
            )
        return procs
    except Exception:
        return {}


def _top_processes(limit: int = 20) -> list[dict]:
    """Return top processes by CPU usage."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "cmdline"]):
        try:
            info = p.info
            procs.append(
                {
                    "pid": info["pid"],
                    "name": info["name"] or "",
                    "cpu_percent": info["cpu_percent"] or 0.0,
                    "memory_mb": round((info["memory_info"].rss if info["memory_info"] else 0) / 1024**2),
                    "command": " ".join(info["cmdline"][:5]) if info["cmdline"] else info["name"] or "",
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: x["cpu_percent"], reverse=True)
    return procs[:limit]


@router.get("/stats")
async def get_system_stats():
    """Return real-time system resource usage."""
    # GPU stats via nvidia-smi
    gpus = _gpu_stats_nvidia_smi() or []

    # Attach per-GPU process info
    if gpus:
        gpu_procs = _gpu_processes()
        for gpu in gpus:
            gpu["processes"] = gpu_procs.get(gpu["index"], [])

    # Add memory warning flags
    from src.services.gpu_monitor import DEFAULT_RESERVED_GB
    for gpu in gpus:
        free_gb = gpu["memory_free_mb"] / 1024
        gpu["low_memory"] = free_gb < DEFAULT_RESERVED_GB

    # Add loaded models to GPU info
    from src.services import model_scheduler
    from src.config import load_model_configs

    configs = load_model_configs()
    loaded = model_scheduler.get_status()["loaded"]
    for model_key in loaded:
        cfg = configs.get(model_key, {})
        gpu_idx = cfg.get("gpu")
        if isinstance(gpu_idx, int):
            # Find the GPU in our list and add the model
            for g in gpus:
                if g["index"] == gpu_idx:
                    if "loaded_models" not in g:
                        g["loaded_models"] = []
                    g["loaded_models"].append({
                        "name": model_key,
                        "type": cfg.get("type", ""),
                        "vram_gb": cfg.get("vram_gb", 0),
                    })

    # CPU stats
    cpu_pct = psutil.cpu_percent(interval=0.1)
    cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)

    # Memory stats
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disk stats
    disk = psutil.disk_usage("/")

    # Uptime
    uptime_seconds = time.time() - psutil.boot_time()

    # Top processes
    processes = _top_processes()

    return {
        "gpus": {"count": len(gpus), "gpus": gpus},
        "system": {
            "cpu_usage_percent": cpu_pct,
            "cpu_count": psutil.cpu_count(),
            "cpu_per_core": cpu_per_core,
            "memory_total_gb": round(mem.total / 1024**3, 1),
            "memory_used_gb": round(mem.used / 1024**3, 1),
            "memory_available_gb": round(mem.available / 1024**3, 1),
            "swap_total_gb": round(swap.total / 1024**3, 1),
            "swap_used_gb": round(swap.used / 1024**3, 1),
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "disk_used_gb": round(disk.used / 1024**3, 1),
            "disk_percent": disk.percent,
        },
        "processes": processes,
        "uptime_seconds": int(uptime_seconds),
    }
