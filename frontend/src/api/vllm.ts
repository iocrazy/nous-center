import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export type VLLMConfig = {
  block_size?: string
  cache_dtype?: string
  enable_prefix_caching?: string
  gpu_memory_utilization?: string
  num_gpu_blocks?: string
}

export type VLLMStats = {
  running?: number
  waiting?: number
  kv_cache_usage_perc?: number
  prefix_cache_queries_total?: number
  prefix_cache_hits_total?: number
  prefix_cache_hit_rate?: number
  num_preemptions_total?: number
}

export type VLLMInstance = {
  name: string
  port: number
  healthy: boolean
  config: VLLMConfig
  stats: VLLMStats
  error: string | null
}

export type VLLMSnapshot = { instances: VLLMInstance[] }

export function useVLLMMetrics(opts?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: ['observability', 'vllm'],
    queryFn: () => apiFetch<VLLMSnapshot>('/api/v1/observability/vllm'),
    // Default 3s — KV usage moves fast under load. Caller can override.
    refetchInterval: opts?.refetchInterval ?? 3_000,
    staleTime: 0,
    retry: false,
  })
}

export type LaunchParamsBody = {
  enable_prefix_caching?: boolean | null
  max_num_seqs?: number | null
  max_model_len?: number | null
  gpu_memory_utilization?: number | null
  tensor_parallel_size?: number | null
  quantization?: string | null
  dtype?: string | null
}

export function useUpdateLaunchParams() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (params: { name: string; body: LaunchParamsBody }) =>
      apiFetch<{ name: string; params: object; applied: boolean; hint: string }>(
        `/api/v1/engines/${encodeURIComponent(params.name)}/launch-params`,
        { method: 'PATCH', body: JSON.stringify(params.body) },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['engines'] })
    },
  })
}
