import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface NodePackageInfo {
  name: string
  version: string
  description: string
  node_count: number
  nodes: string[]
}

export function useNodePackages() {
  return useQuery({
    queryKey: ['node-packages'],
    queryFn: () => apiFetch<Record<string, NodePackageInfo>>('/api/v1/nodes/packages'),
  })
}

export function useNodeDefinitions() {
  return useQuery({
    queryKey: ['node-definitions'],
    queryFn: () => apiFetch<Record<string, unknown>>('/api/v1/nodes/definitions'),
  })
}

export function useRescanPackages() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<{ count: number; packages: string[] }>('/api/v1/nodes/scan', { method: 'POST' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['node-packages'] })
      qc.invalidateQueries({ queryKey: ['node-definitions'] })
    },
  })
}
