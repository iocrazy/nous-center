import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface AgentSummary {
  name: string
  display_name: string
  status: string
  skills: string[]
  model: { engine_key: string | null; fallback_api: string | null }
}

export interface AgentFull extends AgentSummary {
  tools_policy: Record<string, unknown>
  prompts: Record<string, string>
}

export function useAgents() {
  return useQuery({
    queryKey: ['agents'],
    queryFn: () => apiFetch<AgentSummary[]>('/api/v1/agents'),
  })
}

export function useAgent(name: string | null) {
  return useQuery({
    queryKey: ['agent', name],
    queryFn: () => apiFetch<AgentFull>(`/api/v1/agents/${name}`),
    enabled: !!name,
  })
}

export function useCreateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; display_name?: string }) =>
      apiFetch('/api/v1/agents', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}

export function useUpdateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, ...data }: { name: string; display_name?: string; skills?: string[]; model?: Record<string, unknown>; status?: string }) =>
      apiFetch(`/api/v1/agents/${name}`, { method: 'PATCH', body: JSON.stringify(data) }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['agent', vars.name] })
      qc.invalidateQueries({ queryKey: ['agents'] })
    },
  })
}

export function useDeleteAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/agents/${name}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}

export function useSavePrompt() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, filename, content }: { name: string; filename: string; content: string }) =>
      apiFetch(`/api/v1/agents/${name}/prompts/${filename}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: content,
      }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['agent', vars.name] }),
  })
}
