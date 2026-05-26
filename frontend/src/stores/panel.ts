import { create } from 'zustand'

export type PanelId = 'nodes' | 'workflows' | 'presets'
export type OverlayId =
  | 'dashboard'
  | 'models'           // 引擎库
  | 'settings'
  | 'preset-detail'
  | 'api-keys-list'    // m10 v3 API Key 列表
  | 'api-key-detail'   // m10 详情（id 从 URL 取）
  | 'agents'           // legacy — kept addressable for now, hidden in v3 rail
  | 'logs'
  | 'node-packages'    // legacy — moved into Settings sub-page in v3
  | 'services'         // v3 服务列表
  | 'apps'             // legacy alias of services; kept so old links/state don't 404
  | 'service-detail'   // v3 服务详情（id 从 URL 取）
  | 'workflows-list'   // v3 m08 列表（vs canvas at /workflows/:id）
  | 'usage'            // v3 新

/** TaskPanel 模式:dock=右侧抽屉(全高,modal-ish),float=右下角浮窗(可拖拽,不阻塞操作)。
 *  对齐 ComfyUI「停靠 / 悬浮」两态。localStorage 持久。 */
export type TaskPanelMode = 'dock' | 'float'

interface PanelState {
  activePanel: PanelId | null
  activeOverlay: OverlayId | null
  selectedPresetId: string | null
  panelWidth: number
  taskPanelMode: TaskPanelMode
  setPanel: (id: PanelId | null) => void
  togglePanel: (id: PanelId) => void
  setOverlay: (id: OverlayId | null) => void
  toggleOverlay: (id: OverlayId) => void
  openPresetDetail: (presetId: string) => void
  setPanelWidth: (width: number) => void
  setTaskPanelMode: (mode: TaskPanelMode) => void
}

const TASK_PANEL_MODE_KEY = 'nous.taskPanel.mode'
const _initialTaskPanelMode: TaskPanelMode =
  typeof localStorage !== 'undefined' && localStorage.getItem(TASK_PANEL_MODE_KEY) === 'float'
    ? 'float'
    : 'dock'

export const usePanelStore = create<PanelState>((set, get) => ({
  activePanel: 'nodes',
  activeOverlay: null,
  selectedPresetId: null,
  panelWidth: 260,
  taskPanelMode: _initialTaskPanelMode,

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

  setPanelWidth: (width) => set({ panelWidth: Math.max(200, Math.min(400, width)) }),

  setTaskPanelMode: (mode) => {
    if (typeof localStorage !== 'undefined') localStorage.setItem(TASK_PANEL_MODE_KEY, mode)
    set({ taskPanelMode: mode })
  },
}))
