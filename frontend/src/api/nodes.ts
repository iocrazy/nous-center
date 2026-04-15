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


export function useInstallPackageZip() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ file, name }: { file: File; name?: string }) => {
      const fd = new FormData()
      fd.append('file', file)
      if (name) fd.append('name', name)
      return apiFetch<{ installed: string; package_count: number }>(
        '/api/v1/nodes/packages/install_zip',
        { method: 'POST', body: fd },
      )
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['node-packages'] })
      qc.invalidateQueries({ queryKey: ['node-definitions'] })
    },
  })
}


export function useInstallPackageGit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ repo_url, name }: { repo_url: string; name?: string }) => {
      const fd = new FormData()
      fd.append('repo_url', repo_url)
      if (name) fd.append('name', name)
      return apiFetch<{ installed: string; package_count: number }>(
        '/api/v1/nodes/packages/install_git',
        { method: 'POST', body: fd },
      )
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['node-packages'] })
      qc.invalidateQueries({ queryKey: ['node-definitions'] })
    },
  })
}


export function useUninstallPackage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<{ uninstalled: string; package_count: number }>(
        `/api/v1/nodes/packages/${encodeURIComponent(name)}`,
        { method: 'DELETE' },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['node-packages'] })
      qc.invalidateQueries({ queryKey: ['node-definitions'] })
    },
  })
}


export function useInstallPackageDeps() {
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<{ name: string; status: string; log: string }>(
        `/api/v1/nodes/packages/${encodeURIComponent(name)}/install_deps`,
        { method: 'POST' },
      ),
  })
}
