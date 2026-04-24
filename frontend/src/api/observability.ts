import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface GzipSnapshot {
  calls: number
  raw_bytes: number
  compressed_bytes: number
  compression_ratio: number | null
}

export interface CompactionSnapshot {
  calls: number
  turns_dropped: number
  avg_turns_dropped: number | null
  truncated: number
}

export interface CacheSnapshot {
  lookups: number
  hits: number
  hit_rate: number | null
}

export interface RuntimeSnapshot {
  gzip: GzipSnapshot
  compaction: CompactionSnapshot
  cache: CacheSnapshot
}

export function useRuntimeMetrics() {
  return useQuery<RuntimeSnapshot>({
    queryKey: ['runtime-metrics'],
    queryFn: () => apiFetch('/api/v1/observability/runtime'),
    refetchInterval: 30_000,
  })
}
