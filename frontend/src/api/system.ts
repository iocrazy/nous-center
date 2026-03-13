import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

// --- Python backend GPU summary (basic info) ---
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

// --- Rust sidecar endpoints ---

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
}

export interface SysGpuResponse {
  count: number
  gpus: SysGpuInfo[]
}

// Slow down polling when the service is unreachable (error → 10s, ok → normal interval)
function errorAwareInterval(normalMs: number) {
  return (query: { state: { status: string } }) =>
    query.state.status === 'error' ? 10_000 : normalMs
}

export function useSysGpus() {
  return useQuery({
    queryKey: ['sys-gpus'],
    queryFn: () => apiFetch<SysGpuResponse>('/sys/gpus'),
    refetchInterval: errorAwareInterval(2000),
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: false,
  })
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
}

export function useSysStats() {
  return useQuery({
    queryKey: ['sys-stats'],
    queryFn: () => apiFetch<SystemStats>('/sys/stats'),
    refetchInterval: errorAwareInterval(3000),
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export interface ProcessInfo {
  pid: number
  name: string
  cpu_percent: number
  memory_mb: number
  command: string
}

export function useSysProcesses() {
  return useQuery({
    queryKey: ['sys-processes'],
    queryFn: () => apiFetch<{ processes: ProcessInfo[] }>('/sys/processes'),
    refetchInterval: errorAwareInterval(5000),
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: false,
    retry: false,
  })
}
