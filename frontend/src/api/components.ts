import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { apiFetch } from './client'
import { useLiveChannel } from './useLiveChannel'

export interface ComponentInfo {
  filename: string
  abs_path: string
  size_mb: number
  quant_type: string
  mtime: number
}

export type ComponentRole = 'diffusion_models' | 'clip' | 'vae' | 'loras' | 'checkpoint'
export type ComponentLoadState = 'cold' | 'loading' | 'loaded' | 'failed'

/** Same format as backend component_state_key: file|device|dtype|lora_sig.
 * Base loader nodes carry no loras → trailing '|'. */
export function componentStateKey(d: { file?: string; device?: string; dtype?: string }): string {
  return `${d.file ?? ''}|${d.device ?? ''}|${d.dtype ?? ''}|`
}

export function useComponents(role: ComponentRole) {
  return useQuery({
    queryKey: ['components', role],
    queryFn: async () =>
      (await apiFetch<{ components: ComponentInfo[] }>(`/api/v1/components?role=${role}`))
        .components,
    staleTime: 30_000,
  })
}

export interface Seedvr2DitModel {
  filename: string
  label: string
  desc: string
  present: boolean // 磁盘上是否已有(否则选了 NumZ 从 HF 自动下)
  size_mb: number | null
  is_default: boolean
}

/** SeedVR2 DiT 白名单 + 磁盘状态。给 seedvr2_model_select widget「混合」展示
 *  (盘上有标已就绪 / 白名单其余标可下载)。 */
export function useSeedvr2DitModels() {
  return useQuery({
    queryKey: ['seedvr2-dit'],
    queryFn: async () =>
      (await apiFetch<{ models: Seedvr2DitModel[] }>('/api/v1/components/seedvr2-dit')).models,
    staleTime: 30_000,
  })
}

interface ComponentStateStore {
  states: Record<string, { state: ComponentLoadState; error: string | null }>
  set: (key: string, state: ComponentLoadState, error: string | null) => void
}
export const useComponentStateStore = create<ComponentStateStore>((set) => ({
  states: {},
  set: (key, state, error) =>
    set((s) => ({ states: { ...s.states, [key]: { state, error } } })),
}))

/** Subscribe /ws/models for component_state_changed → update the store. */
export function useComponentStateLiveSync(): void {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`
  const setState = useComponentStateStore((s) => s.set)
  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event === 'component_state_changed') {
        setState(
          String(data.component_key),
          (data.state as ComponentLoadState) ?? 'cold',
          (data.error as string) ?? null,
        )
      }
    },
  })
}

/** All currently-tracked component load-states (runner L1 mirror). Live via
 * /ws/models invalidation. Used by the service overview to ask "is this
 * component file loaded?" — matched by file (state-key's first segment) so
 * it's robust to device/dtype/lora resolution. Unlisted files → cold. */
export function useAllComponentStates() {
  const qc = useQueryClient()
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`
  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event === 'component_state_changed') {
        qc.invalidateQueries({ queryKey: ['component-states-all'] })
      }
    },
    onReconnect: () => qc.invalidateQueries({ queryKey: ['component-states-all'] }),
  })
  return useQuery({
    queryKey: ['component-states-all'],
    queryFn: async () =>
      (await apiFetch<{ components: { key: string; state: ComponentLoadState; error: string | null }[] }>(
        '/api/v1/models/components/state',
      )).components,
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
}

/** Map component file → best load-state across all (device/dtype/lora) variants.
 * loaded > loading > failed > cold. The state-key format is file|device|dtype|lora. */
export function loadedStateByFile(
  states: { key: string; state: ComponentLoadState }[] | undefined,
): Record<string, ComponentLoadState> {
  const rank: Record<ComponentLoadState, number> = { loaded: 3, loading: 2, failed: 1, cold: 0 }
  const out: Record<string, ComponentLoadState> = {}
  for (const s of states ?? []) {
    const file = s.key.split('|')[0]
    if (!file) continue
    if (!out[file] || rank[s.state] > rank[out[file]]) out[file] = s.state
  }
  return out
}

/** Live four-state for one component key. Seeds from GET /state on mount, then
 * tracks WS pushes. Unknown → 'cold'. */
export function useComponentState(key: string): { state: ComponentLoadState; error: string | null } {
  useComponentStateLiveSync()
  const qc = useQueryClient()
  const setState = useComponentStateStore((s) => s.set)
  const entry = useComponentStateStore((s) => s.states[key])
  useEffect(() => {
    if (!key.replace(/\|/g, '')) return
    let cancelled = false
    apiFetch<{ components: { key: string; state: ComponentLoadState; error: string | null }[] }>(
      `/api/v1/models/components/state?keys=${encodeURIComponent(key)}`,
    )
      .then((resp) => {
        if (cancelled) return
        const row = resp.components.find((r) => r.key === key)
        if (row) setState(key, row.state, row.error)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [key, setState, qc])
  return entry ?? { state: 'cold', error: null }
}
