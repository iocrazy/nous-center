import { create } from 'zustand'

export type PanelId = 'nodes' | 'workflows' | 'presets' | 'collections' | 'api'
export type OverlayId = 'dashboard' | 'models' | 'settings' | 'preset-detail' | 'instance-detail'

interface PanelState {
  activePanel: PanelId | null
  activeOverlay: OverlayId | null
  selectedPresetId: string | null
  selectedInstanceId: string | null
  panelWidth: number
  setPanel: (id: PanelId | null) => void
  togglePanel: (id: PanelId) => void
  setOverlay: (id: OverlayId | null) => void
  toggleOverlay: (id: OverlayId) => void
  openPresetDetail: (presetId: string) => void
  openInstanceDetail: (instanceId: string) => void
  setPanelWidth: (width: number) => void
}

export const usePanelStore = create<PanelState>((set, get) => ({
  activePanel: 'nodes',
  activeOverlay: null,
  selectedPresetId: null,
  selectedInstanceId: null,
  panelWidth: 260,

  setPanel: (id) => set({ activePanel: id, activeOverlay: null }),

  togglePanel: (id) => {
    const { activePanel } = get()
    set({
      activePanel: activePanel === id ? null : id,
      activeOverlay: null,
    })
  },

  setOverlay: (id) => set({ activeOverlay: id, activePanel: null }),

  toggleOverlay: (id) => {
    const { activeOverlay } = get()
    set({
      activeOverlay: activeOverlay === id ? null : id,
      activePanel: null,
    })
  },

  openPresetDetail: (presetId) =>
    set({ activeOverlay: 'preset-detail', activePanel: null, selectedPresetId: presetId }),

  openInstanceDetail: (instanceId) =>
    set({ activeOverlay: 'instance-detail', activePanel: null, selectedInstanceId: instanceId }),

  setPanelWidth: (width) => set({ panelWidth: Math.max(200, Math.min(400, width)) }),
}))
