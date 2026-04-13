# GPU Process Management

## Problem

When the backend restarts, previously spawned vLLM subprocesses become orphans — they hold GPU memory but ModelManager has no record of them. The Dashboard shows "Loaded Models (0)" while GPUs report 14-15GB occupied. Users have no way to see or kill these orphan processes from the UI.

## Goals

1. **Visibility**: Show per-GPU processes in the Dashboard with orphan/managed status
2. **Manual control**: Kill button for orphan GPU processes from the UI
3. **Auto-recovery on startup**: Reconnect healthy orphan vLLM processes, kill unhealthy ones

## Non-Goals

- Automatic periodic orphan scanning (only on startup + manual via UI)

---

## Backend Changes

### 1. Enhance `_gpu_processes()` in `monitor.py`

Current return: `{pid, used_gpu_memory_mb}`

New return per process:
```python
{
    "pid": int,
    "gpu": int,                   # GPU index
    "used_gpu_memory_mb": int,
    "name": str,                  # process name (e.g. "python")
    "command": str,               # truncated cmdline (first 120 chars)
    "managed": bool,              # True if tracked by ModelManager
    "model_name": str | None,     # model ID if managed, else None
}
```

Implementation: After collecting PIDs from nvidia-smi, use `psutil.Process(pid)` to read name and cmdline. Cross-reference with ModelManager's loaded models to determine managed status via `get_pid_map()`.

Note: `_gpu_processes()` is sync and called from async handler. Keep psutil lookups lightweight (no blocking I/O). If performance becomes an issue, wrap in `run_in_executor`.

### 2. New endpoint: `POST /api/v1/monitor/kill-process`

Request body:
```python
class KillProcessRequest(BaseModel):
    pid: int
```

Logic:
1. Verify the PID exists in the current GPU process list (safety: only kill GPU-using processes)
2. Verify the process is NOT managed by ModelManager (prevent killing active models — use unload API instead)
3. Kill with `os.kill(pid, SIGTERM)`, wait up to 5s, fallback to `os.kill(pid, SIGKILL)`. Use single-process kill (NOT `os.killpg`) because orphan processes may share a process group with the user's shell
4. Return `{"killed": true, "pid": pid}`

Error cases:
- PID not found in GPU processes -> 404
- PID is managed by ModelManager -> 409 Conflict ("Use the unload API instead")
- Kill fails -> 500

### 3. Enhance startup orphan handling in `main.py` lifespan

Current flow (vllm_scanner): scan processes -> health check -> only return healthy ones -> reconnect.

Enhanced flow:
1. `scan_running_vllm()` returns ALL vLLM processes with a `healthy: bool` field
2. Main loop separates them:
   - **Healthy**: reconnect to ModelManager (existing logic)
   - **Unhealthy**: kill via `os.kill(pid, SIGTERM)` and log
3. Log each action clearly

Important: unhealthy processes must NOT be passed to `model_mgr.load_model()`, which would trigger `VLLMAdapter.load()` falling through to spawn a duplicate subprocess.

### 4. ModelManager: add PID tracking

Add method to ModelManager:

```python
def get_pid_map(self) -> dict[int, str]:
    """Return {pid: model_id} for all managed processes that have a PID."""
```

Implementation: iterate `self._models`, for each adapter check:
- `adapter._process.pid` if `adapter._process` is not None (subprocess we spawned)
- `getattr(adapter, '_adopted_pid', None)` (reconnected orphan — attribute only set in `load()`)

---

## Frontend Changes

### 1. GPU Process list inside GpuPanel

Currently `gpu.processes` data flows to the frontend but is not rendered. Add a process list below the MEM bar in each `GpuPanel`. Each row shows:

```
PID       MEM      COMMAND                              ACTION
1126467   14.7G    vllm...Qwen3.5-35B-A3B-GPTQ...      [Kill]
```

- **Managed processes**: muted text, no kill button, show model name instead of command
- **Orphan processes**: warning color, kill button
- Only render if `gpu.processes` has entries

### 2. Kill mutation

Add `useKillProcess()` hook in the system API module:
```typescript
useMutation({
    mutationFn: (pid: number) => apiFetch('/api/v1/monitor/kill-process', {
        method: 'POST', body: JSON.stringify({ pid }),
    }),
    onSuccess: () => {
        qc.invalidateQueries({ queryKey: ['monitor-stats'] })
    },
})
```

### 3. Kill confirmation

`window.confirm("Kill process PID {pid}? This will free ~{mem}G GPU memory.")`

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/src/api/routes/monitor.py` | Enhance `_gpu_processes()`, add kill endpoint |
| `backend/src/services/model_manager.py` | Add `get_pid_map()` method |
| `backend/src/services/inference/llm_vllm.py` | Expose PID via property |
| `backend/src/services/inference/vllm_scanner.py` | Return all processes with `healthy` flag |
| `backend/src/api/main.py` | Kill unhealthy orphans on startup |
| `frontend/src/api/system.ts` | Add `useKillProcess()` hook, update process types |
| `frontend/src/components/overlays/DashboardOverlay.tsx` | GPU process list in GpuPanel |

## Testing

- Start vLLM manually, restart backend -> should reconnect
- Start vLLM, kill backend, corrupt vLLM (block its port) -> should auto-kill on restart
- Dashboard should show managed processes without kill button
- Dashboard should show orphan processes with kill button
- Killing orphan should free GPU memory and update display
