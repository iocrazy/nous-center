import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

// --- Python backend GPU summary (basic info, used by engines page) ---
export interface GpuInfo {
  index: number
  name: string
  vram_total_gb: number
  compute_capability: [number, number]
}

export interface GpuSummary {
  count: number
  gpus: GpuInfo[]
}

export function useGpuStats() {
  return useQuery({
    queryKey: ['gpu-stats'],
    queryFn: () => apiFetch<GpuSummary>('/api/v1/engines/gpus'),
    refetchInterval: 3000,
  })
}

// --- Unified monitor endpoint (replaces Rust sidecar) ---

export interface SysGpuInfo {
  index: number
  name: string
  utilization_gpu: number
  utilization_memory: number
  temperature: number
  fan_speed: number
  power_draw_w: number
  power_limit_w: number
  memory_used_mb: number
  memory_total_mb: number
  memory_free_mb: number
  processes: { pid: number; used_gpu_memory_mb: number }[]
  loaded_models?: { name: string; type: string; vram_gb: number }[]
  low_memory?: boolean
}

export interface SysGpuResponse {
  count: number
  gpus: SysGpuInfo[]
}

export interface SystemStats {
  cpu_usage_percent: number
  cpu_count: number
  cpu_per_core: number[]
  memory_total_gb: number
  memory_used_gb: number
  memory_available_gb: number
  swap_total_gb: number
  swap_used_gb: number
  disk_total_gb: number
  disk_used_gb: number
  disk_percent: number
}

export interface ProcessInfo {
  pid: number
  name: string
  cpu_percent: number
  memory_mb: number
  command: string
}

interface MonitorStatsResponse {
  gpus: SysGpuResponse
  system: SystemStats
  processes: ProcessInfo[]
  uptime_seconds: number
}

// Single query that fetches everything from the Python backend
export function useMonitorStats() {
  return useQuery({
    queryKey: ['monitor-stats'],
    queryFn: () => apiFetch<MonitorStatsResponse>('/api/v1/monitor/stats'),
    refetchInterval: 2000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
  })
}

// Convenience hooks that derive from the unified query
export function useSysGpus() {
  const q = useMonitorStats()
  return { ...q, data: q.data?.gpus }
}

export function useSysStats() {
  const q = useMonitorStats()
  return { ...q, data: q.data?.system }
}

export function useSysProcesses() {
  const q = useMonitorStats()
  return { ...q, data: q.data ? { processes: q.data.processes } : undefined }
}
