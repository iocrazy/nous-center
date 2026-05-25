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

export type ComponentRole = 'diffusion_models' | 'clip' | 'vae' | 'loras'
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
