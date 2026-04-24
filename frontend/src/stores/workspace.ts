import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { Workflow, WorkflowNode, WorkflowEdge } from '../models/workflow'
import { uid } from '../utils/uid'
import { apiFetch } from '../api/client'
import type { WorkflowFull } from '../api/workflows'

let saveTimer: ReturnType<typeof setTimeout> | null = null

const HISTORY_LIMIT = 50

export interface WorkflowTab {
  id: string
  name: string
  workflow: Workflow
  isDirty: boolean
  dbId: string | null
  /** Undo stack (past snapshots of `workflow`). Newest at the end. */
  past: Workflow[]
  /** Redo stack (snapshots undone from current). Newest at the end. */
  future: Workflow[]
}

function snapshot(wf: Workflow): Workflow {
  // Shallow clone of the arrays — node/edge objects themselves are treated as
  // immutable by every mutation below, so we don't need a deep clone.
  return { ...wf, nodes: [...wf.nodes], edges: [...wf.edges] }
}

function pushHistory(tab: WorkflowTab): WorkflowTab {
  const past = [...tab.past, snapshot(tab.workflow)]
  if (past.length > HISTORY_LIMIT) past.shift()
  return { ...tab, past, future: [] }
}

function createDefaultWorkflow(name: string): Workflow {
  return {
    id: uid(),
    name,
    nodes: [],
    edges: [],
  }
}

function createTab(name: string, workflow: Workflow, dbId: string | null = null): WorkflowTab {
  return {
    id: uid(),
    name,
    workflow,
    isDirty: false,
    dbId,
    past: [],
    future: [],
  }
}

interface WorkspaceState {
  tabs: WorkflowTab[]
  activeTabId: string

  // Tab management
  addTab: (name?: string) => void
  removeTab: (id: string) => void
  setActiveTab: (id: string) => void
  renameTab: (id: string, name: string) => void

  // DB integration
  loadFromDb: (workflow: WorkflowFull) => void

  // Workflow mutations (on active tab)
  getActiveWorkflow: () => Workflow
  setWorkflow: (wf: Workflow) => void
  updateNode: (nodeId: string, data: Record<string, unknown>) => void
  addNode: (node: WorkflowNode) => void
  removeNode: (nodeId: string) => void
  addEdge: (edge: WorkflowEdge) => void
  removeEdge: (edgeId: string) => void
  markDirty: () => void

  // Undo / redo (structural changes only — add/remove node + edge)
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean

  /** Activate an existing tab by its DB-backed workflow id. Returns true if
   *  found; false means caller should fetch from backend + loadFromDb(). */
  activateByDbId: (dbId: string) => boolean
}

const initialTab: WorkflowTab = createTab('新工作流', createDefaultWorkflow('基础合成'))

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
  tabs: [initialTab],
  activeTabId: initialTab.id,

  addTab: (name) => {
    const tabName = name ?? `工作流 ${get().tabs.length + 1}`
    const tab = createTab(tabName, createDefaultWorkflow(tabName))
    set((s) => ({ tabs: [...s.tabs, tab], activeTabId: tab.id }))
  },

  removeTab: (id) => {
    const { tabs, activeTabId } = get()
    if (tabs.length <= 1) return
    const idx = tabs.findIndex((t) => t.id === id)
    const newTabs = tabs.filter((t) => t.id !== id)
    const newActive = id === activeTabId
      ? newTabs[Math.min(idx, newTabs.length - 1)].id
      : activeTabId
    set({ tabs: newTabs, activeTabId: newActive })
  },

  setActiveTab: (id) => set({ activeTabId: id }),

  renameTab: (id, name) => {
    const tab = get().tabs.find((t) => t.id === id)
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === id ? { ...t, name, workflow: { ...t.workflow, name } } : t
      ),
    }))
    // Sync name to backend
    if (tab?.dbId) {
      apiFetch(`/api/v1/workflows/${tab.dbId}`, {
        method: 'PATCH',
        body: JSON.stringify({ name }),
      }).catch((err) => console.error('Rename sync failed', err))
    }
  },

  loadFromDb: (wf: WorkflowFull) => {
    const workflow: Workflow = {
      id: wf.id,
      name: wf.name,
      description: wf.description ?? undefined,
      nodes: wf.nodes,
      edges: wf.edges,
      is_template: wf.is_template,
      status: wf.status as 'draft' | 'published',
    }
    const existing = get().tabs.find((t) => t.dbId === wf.id)
    if (existing) {
      // 已有 tab — 用 db 内容刷新（覆盖本地未保存的 stale state）+ 激活
      set((s) => ({
        tabs: s.tabs.map((t) =>
          t.id === existing.id
            ? { ...t, name: wf.name, workflow, isDirty: false }
            : t,
        ),
        activeTabId: existing.id,
      }))
      return
    }
    const tab = createTab(wf.name, workflow, wf.id)
    set((s) => ({ tabs: [...s.tabs, tab], activeTabId: tab.id }))
  },

  getActiveWorkflow: () => {
    const { tabs, activeTabId } = get()
    return tabs.find((t) => t.id === activeTabId)!.workflow
  },

  setWorkflow: (wf) =>
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId ? { ...t, workflow: wf, isDirty: true } : t
      ),
    })),

  updateNode: (nodeId, data) => {
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...t,
              isDirty: true,
              workflow: {
                ...t.workflow,
                nodes: t.workflow.nodes.map((n) =>
                  n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n
                ),
              },
            }
          : t
      ),
    }))
    get().markDirty()
  },

  addNode: (node) => {
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...pushHistory(t),
              isDirty: true,
              workflow: { ...t.workflow, nodes: [...t.workflow.nodes, node] },
            }
          : t
      ),
    }))
    get().markDirty()
  },

  removeNode: (nodeId) => {
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...pushHistory(t),
              isDirty: true,
              workflow: {
                ...t.workflow,
                nodes: t.workflow.nodes.filter((n) => n.id !== nodeId),
                edges: t.workflow.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
              },
            }
          : t
      ),
    }))
    get().markDirty()
  },

  addEdge: (edge) => {
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...pushHistory(t),
              isDirty: true,
              workflow: { ...t.workflow, edges: [...t.workflow.edges, edge] },
            }
          : t
      ),
    }))
    get().markDirty()
  },

  removeEdge: (edgeId) => {
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...pushHistory(t),
              isDirty: true,
              workflow: {
                ...t.workflow,
                edges: t.workflow.edges.filter((e) => e.id !== edgeId),
              },
            }
          : t
      ),
    }))
    get().markDirty()
  },

  undo: () => {
    const t = get().tabs.find((x) => x.id === get().activeTabId)
    if (!t || t.past.length === 0) return
    const prev = t.past[t.past.length - 1]
    set((s) => ({
      tabs: s.tabs.map((x) =>
        x.id === s.activeTabId
          ? {
              ...x,
              past: x.past.slice(0, -1),
              future: [...x.future, snapshot(x.workflow)],
              workflow: prev,
              isDirty: true,
            }
          : x,
      ),
    }))
    get().markDirty()
  },

  redo: () => {
    const t = get().tabs.find((x) => x.id === get().activeTabId)
    if (!t || t.future.length === 0) return
    const next = t.future[t.future.length - 1]
    set((s) => ({
      tabs: s.tabs.map((x) =>
        x.id === s.activeTabId
          ? {
              ...x,
              past: [...x.past, snapshot(x.workflow)],
              future: x.future.slice(0, -1),
              workflow: next,
              isDirty: true,
            }
          : x,
      ),
    }))
    get().markDirty()
  },

  canUndo: () => {
    const t = get().tabs.find((x) => x.id === get().activeTabId)
    return !!t && t.past.length > 0
  },

  canRedo: () => {
    const t = get().tabs.find((x) => x.id === get().activeTabId)
    return !!t && t.future.length > 0
  },

  activateByDbId: (dbId) => {
    const tab = get().tabs.find((t) => t.dbId === dbId)
    if (!tab) return false
    set({ activeTabId: tab.id })
    return true
  },

  markDirty: (tabId?: string) => {
    const activeTabId = tabId ?? get().activeTabId
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId ? { ...t, isDirty: true } : t
      ),
    }))

    if (saveTimer) clearTimeout(saveTimer)
    saveTimer = setTimeout(async () => {
      const current = get().tabs.find((t) => t.id === activeTabId)
      if (!current?.workflow) return

      try {
        if (current.dbId) {
          // Existing workflow — PATCH
          await apiFetch(`/api/v1/workflows/${current.dbId}`, {
            method: 'PATCH',
            body: JSON.stringify({ name: current.name, nodes: current.workflow.nodes, edges: current.workflow.edges }),
          })
          set((s) => ({
            tabs: s.tabs.map((t) =>
              t.id === activeTabId ? { ...t, isDirty: false } : t
            ),
          }))
        } else {
          // New workflow — POST to create, then store dbId
          const created = await apiFetch<{ id: string }>('/api/v1/workflows', {
            method: 'POST',
            body: JSON.stringify({
              name: current.workflow.name || current.name,
              nodes: current.workflow.nodes,
              edges: current.workflow.edges,
            }),
          })
          set((s) => ({
            tabs: s.tabs.map((t) =>
              t.id === activeTabId ? { ...t, dbId: created.id, isDirty: false } : t
            ),
          }))
        }
      } catch (err) {
        console.error('Auto-save failed', err)
      }
    }, 2000)
  },
    }),
    {
      name: 'nous-center:workspace',
      storage: createJSONStorage(() => localStorage),
      version: 1,
      // 只持久化 tab 元信息 + 当前 active；workflow content / past / future 太大，
      // 而且 dbId 已存的 tab 下次激活时 WorkflowRouteLoader 会按需 fetch（旧版没
      // 走这条路所以 unsaved 内容会丢，但本地编辑已有 2s 自动 save 兜底）。
      partialize: (state) => ({
        // 丢空白 unsaved tab — 它们就是默认创建的"新工作流"，重启后没保留意义
        tabs: state.tabs
          .filter((t) => !!t.dbId || t.workflow.nodes.length > 0 || t.workflow.edges.length > 0)
          .slice(-10) // 最多记 10 个，防止 localStorage 膨胀
          .map((t) => ({
            id: t.id,
            name: t.name,
            workflow: t.workflow,
            isDirty: t.isDirty,
            dbId: t.dbId,
            // past / future 不持久化 — undo 历史跨刷新无意义且占空间
            past: [] as Workflow[],
            future: [] as Workflow[],
          })),
        activeTabId: state.activeTabId,
      }),
      // hydrate 后保底：如果反序列化出来 tabs 为空（首次 / 全部被过滤），
      // 用默认 initialTab 顶上避免空 state。activeTabId 也校正到第一个有效 tab。
      onRehydrateStorage: () => (state) => {
        if (!state) return
        if (state.tabs.length === 0) {
          state.tabs = [initialTab]
          state.activeTabId = initialTab.id
        } else if (!state.tabs.some((t) => t.id === state.activeTabId)) {
          state.activeTabId = state.tabs[0].id
        }
      },
    },
  ),
)
