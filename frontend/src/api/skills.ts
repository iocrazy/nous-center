import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface SkillSummary {
  name: string
  description: string
  requires: Record<string, unknown>
  dir_name: string
}

export interface SkillFull extends SkillSummary {
  body: string
  raw: string
}

export function useSkills() {
  return useQuery({
    queryKey: ['skills'],
    queryFn: () => apiFetch<SkillSummary[]>('/api/v1/skills'),
  })
}

export function useSkill(name: string | null) {
  return useQuery({
    queryKey: ['skill', name],
    queryFn: () => apiFetch<SkillFull>(`/api/v1/skills/${name}`),
    enabled: !!name,
  })
}

export function useCreateSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; description?: string; body?: string }) =>
      apiFetch('/api/v1/skills', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }),
  })
}

export function useUpdateSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, raw }: { name: string; raw: string }) =>
      apiFetch(`/api/v1/skills/${name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: raw,
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['skill', vars.name] })
      qc.invalidateQueries({ queryKey: ['skills'] })
    },
  })
}

export function useDeleteSkill() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/skills/${name}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['skills'] }),
  })
}
