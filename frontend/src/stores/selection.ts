import { create } from 'zustand'

/** m09: 当前画布选中的节点 ID。仅由 ReactFlow 的 onSelectionChange 写入，
 *  NodePropertyPanel 读它决定渲染谁的 widgets。 */
interface SelectionState {
  selectedNodeId: string | null
  setSelectedNodeId: (id: string | null) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
}))
