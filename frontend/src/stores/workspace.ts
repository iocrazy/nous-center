import { create } from 'zustand'
import type { Workflow, WorkflowNode, WorkflowEdge } from '../models/workflow'
import { uid } from '../utils/uid'
import { apiFetch } from '../api/client'
import type { WorkflowFull } from '../api/workflows'

let saveTimer: ReturnType<typeof setTimeout> | null = null

export interface WorkflowTab {
  id: string
  name: string
  workflow: Workflow
  isDirty: boolean
  dbId: string | null
}

function createDefaultWorkflow(name: string): Workflow {
  const textId = uid()
  const engineId = uid()
  const outputId = uid()
  return {
    id: uid(),
    name,
    nodes: [
      { id: textId, type: 'text_input', data: { text: '' }, position: { x: 100, y: 200 } },
      { id: engineId, type: 'tts_engine', data: { engine: 'cosyvoice2' }, position: { x: 400, y: 200 } },
      { id: outputId, type: 'output', data: {}, position: { x: 700, y: 200 } },
    ],
    edges: [
      { id: uid(), source: textId, sourceHandle: 'text', target: engineId, targetHandle: 'text' },
      { id: uid(), source: engineId, sourceHandle: 'audio', target: outputId, targetHandle: 'audio' },
    ],
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
}

const initialTab: WorkflowTab = {
  id: uid(),
  name: '基础合成',
  workflow: createDefaultWorkflow('基础合成'),
  isDirty: false,
  dbId: null,
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  tabs: [initialTab],
  activeTabId: initialTab.id,

  addTab: (name) => {
    const tabName = name ?? `工作流 ${get().tabs.length + 1}`
    const tab: WorkflowTab = {
      id: uid(),
      name: tabName,
      workflow: createDefaultWorkflow(tabName),
      isDirty: false,
      dbId: null,
    }
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

  renameTab: (id, name) =>
    set((s) => ({
      tabs: s.tabs.map((t) => (t.id === id ? { ...t, name } : t)),
    })),

  loadFromDb: (wf: WorkflowFull) => {
    const tab: WorkflowTab = {
      id: uid(),
      name: wf.name,
      workflow: {
        id: wf.id,
        name: wf.name,
        description: wf.description ?? undefined,
        nodes: wf.nodes,
        edges: wf.edges,
        is_template: wf.is_template,
        status: wf.status as 'draft' | 'published',
      },
      isDirty: false,
      dbId: wf.id,
    }
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

  updateNode: (nodeId, data) =>
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
    })),

  addNode: (node) =>
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? { ...t, isDirty: true, workflow: { ...t.workflow, nodes: [...t.workflow.nodes, node] } }
          : t
      ),
    })),

  removeNode: (nodeId) =>
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...t,
              isDirty: true,
              workflow: {
                ...t.workflow,
                nodes: t.workflow.nodes.filter((n) => n.id !== nodeId),
                edges: t.workflow.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
              },
            }
          : t
      ),
    })),

  addEdge: (edge) =>
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? { ...t, isDirty: true, workflow: { ...t.workflow, edges: [...t.workflow.edges, edge] } }
          : t
      ),
    })),

  removeEdge: (edgeId) =>
    set((s) => ({
      tabs: s.tabs.map((t) =>
        t.id === s.activeTabId
          ? {
              ...t,
              isDirty: true,
              workflow: {
                ...t.workflow,
                edges: t.workflow.edges.filter((e) => e.id !== edgeId),
              },
            }
          : t
      ),
    })),

  markDirty: () => {
    const activeTabId = get().activeTabId
    const tab = get().tabs.find((t) => t.id === activeTabId)
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
            body: JSON.stringify({ nodes: current.workflow.nodes, edges: current.workflow.edges }),
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
}))
