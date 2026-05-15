"""Real-time system resource monitoring endpoint."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

import psutil
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from src.api.deps_admin import require_admin

logger = logging.getLogger(__name__)

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


def _gpu_processes(pid_map: dict[int, str] | None = None) -> dict[int, list[dict]]:
    """Get per-GPU process memory usage via nvidia-smi, enriched with process info."""
    if pid_map is None:
        pid_map = {}
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

            pid = int(parts[1])
            mem_mb = int(parts[2])

            # Enrich with process name/command via psutil
            name = ""
            command = ""
            try:
                p = psutil.Process(pid)
                name = p.name()
                cmdline = p.cmdline()
                command = " ".join(cmdline[:8])[:120] if cmdline else name
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            managed = pid in pid_map
            model_name = pid_map.get(pid)

            procs.setdefault(gpu_idx, []).append(
                {
                    "pid": pid,
                    "gpu": gpu_idx,
                    "used_gpu_memory_mb": mem_mb,
                    "name": name,
                    "command": command,
                    "managed": managed,
                    "model_name": model_name,
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
async def get_system_stats(request: Request):
    """Return real-time system resource usage."""
    # GPU stats via nvidia-smi
    gpus = _gpu_stats_nvidia_smi() or []

    # Get PID map from ModelManager for managed/orphan detection
    pid_map: dict[int, str] = {}
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is not None:
        pid_map = model_mgr.get_pid_map()

    # Attach per-GPU process info
    if gpus:
        gpu_procs = _gpu_processes(pid_map)
        for gpu in gpus:
            gpu["processes"] = gpu_procs.get(gpu["index"], [])

    # Add memory warning flags
    from src.services.gpu_monitor import DEFAULT_RESERVED_GB
    for gpu in gpus:
        free_gb = gpu["memory_free_mb"] / 1024
        gpu["low_memory"] = free_gb < DEFAULT_RESERVED_GB

    # Add loaded models to GPU info
    from src.config import load_model_configs
    from src.gpu.detector import get_device_for_engine

    configs = load_model_configs()
    loaded = model_mgr.loaded_model_ids if model_mgr is not None else []
    for model_key in loaded:
        cfg = configs.get(model_key, {})
        gpu_idx = cfg.get("gpu")
        # If the config didn't pin a GPU, resolve via detector (same logic the
        # scheduler used at load time, so the answer matches where the engine
        # actually landed).
        if gpu_idx is None:
            device = get_device_for_engine(cfg)
            if device.startswith("cuda:"):
                try:
                    gpu_idx = int(device.split(":")[-1])
                except ValueError:
                    gpu_idx = None
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


class KillProcessRequest(BaseModel):
    pid: int


@router.post("/kill-process", dependencies=[Depends(require_admin)])
async def kill_gpu_process(req: KillProcessRequest, request: Request):
    """Kill an orphan GPU process by PID. Admin only — can disrupt any GPU worker."""
    pid = req.pid

    # 1. Verify PID is in GPU process list
    gpu_procs = _gpu_processes()
    all_gpu_pids = {p["pid"] for procs in gpu_procs.values() for p in procs}
    if pid not in all_gpu_pids:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"PID {pid} not found in GPU process list"},
        )

    # 2. Verify not managed by ModelManager
    model_mgr = getattr(request.app.state, "model_manager", None)
    if model_mgr is not None:
        pid_map = model_mgr.get_pid_map()
        if pid in pid_map:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=409,
                content={"detail": f"PID {pid} is managed model '{pid_map[pid]}'. Use the unload API instead."},
            )

    # 3. Kill with SIGTERM, fallback to SIGKILL
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to GPU process %d", pid)
        # Wait for process to exit
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # Check if still alive
            except ProcessLookupError:
                return {"killed": True, "pid": pid}
        # Still alive, force kill
        os.kill(pid, signal.SIGKILL)
        logger.info("Sent SIGKILL to GPU process %d", pid)
        return {"killed": True, "pid": pid}
    except ProcessLookupError:
        return {"killed": True, "pid": pid}  # Already dead
    except Exception as e:
        logger.warning("Failed to kill process %d: %s", pid, e)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to kill process {pid}: {e}"},
        )


@router.get("/runners")
async def list_runners(request: Request) -> dict:
    """Per-runner 状态，给前端 TaskPanel 的 Buildkite 风 runner 泳道用。

    数据源 = RunnerSupervisor.health_snapshot()（Lane H Task 4）。
    runner_supervisors 由 scheduler / Lane A 接入填充；未接入时返回空列表。
    """
    supervisors = getattr(request.app.state, "runner_supervisors", [])
    return {"runners": [s.health_snapshot() for s in supervisors]}


@router.get("/usage/summary")
async def usage_summary():
    """Get aggregated usage stats for dashboard."""
    from src.services.usage_service import get_usage_summary
    return await get_usage_summary()


@router.get("/usage/by-model")
async def usage_by_model(since: str | None = None):
    """Get per-model usage breakdown."""
    from src.services.usage_service import get_usage_by_model
    from datetime import datetime
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            pass
    return await get_usage_by_model(since=since_dt)


@router.get("/usage/inference")
async def usage_inference(
    start: str | None = None,
    end: str | None = None,
    interval: str = "day",
    group_by: str = "Model",
    instance_id: int | None = None,
    model: str | None = None,
    format: str = "json",
):
    """Ark-style inference usage query.

    - interval: day | hour
    - group_by: Model | Instance | ApiKey
    - format: json (default, idiomatic list) | columnar (Ark Fields+Data)
    """
    from src.services.usage_service import get_inference_usage
    from datetime import datetime

    def _parse(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    return await get_inference_usage(
        start=_parse(start),
        end=_parse(end),
        interval=interval,
        group_by=group_by,
        instance_id=instance_id,
        model=model,
        columnar=(format == "columnar"),
    )
