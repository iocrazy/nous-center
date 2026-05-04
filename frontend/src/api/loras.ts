import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface LoraInfo {
  name: string
  size_mb: number
  subdir: string
}

export function useLoras() {
  return useQuery({
    queryKey: ['loras'],
    queryFn: () => apiFetch<LoraInfo[]>('/api/v1/loras'),
    staleTime: 30_000,
  })
}
