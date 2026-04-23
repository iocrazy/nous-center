import { create } from 'zustand'

export type PanelId = 'nodes' | 'workflows' | 'presets'
export type OverlayId =
  | 'dashboard'
  | 'models'           // 引擎库
  | 'settings'
  | 'preset-detail'
  | 'api-management'   // API Key
  | 'agents'           // legacy — kept addressable for now, hidden in v3 rail
  | 'logs'
  | 'node-packages'    // legacy — moved into Settings sub-page in v3
  | 'services'         // v3 服务列表
  | 'apps'             // legacy alias of services; kept so old links/state don't 404
  | 'service-detail'   // v3 服务详情（id 从 URL 取）
  | 'usage'            // v3 新

export interface ApiManagementOptions {
  presetId?: string
  instanceId?: string
}

interface PanelState {
  activePanel: PanelId | null
  activeOverlay: OverlayId | null
  selectedPresetId: string | null
  apiManagementOptions: ApiManagementOptions | null
  panelWidth: number
  setPanel: (id: PanelId | null) => void
  togglePanel: (id: PanelId) => void
  setOverlay: (id: OverlayId | null) => void
  toggleOverlay: (id: OverlayId) => void
  openPresetDetail: (presetId: string) => void
  openApiManagement: (options?: ApiManagementOptions) => void
  setPanelWidth: (width: number) => void
}

export const usePanelStore = create<PanelState>((set, get) => ({
  activePanel: 'nodes',
  activeOverlay: null,
  selectedPresetId: null,
  apiManagementOptions: null,
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

  openApiManagement: (options) =>
    set({ activeOverlay: 'api-management', activePanel: null, apiManagementOptions: options ?? null }),

  setPanelWidth: (width) => set({ panelWidth: Math.max(200, Math.min(400, width)) }),
}))
