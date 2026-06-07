/**
 * pasteGraph — 复制粘贴的纯逻辑:给剪贴板里的节点 + 它们之间的内部边分配新 id、
 * 整体偏移落位,内部边按「原 id → 新 id」重连。从 NodeEditor 抽出便于单测(React Flow
 * 框选交互难自动化,但这段确定性 map 逻辑是 paste 的关键,单测锁住回归)。
 */

export interface ClipNode {
  id: string
  type: string
  data: Record<string, unknown>
  position: { x: number; y: number }
  style?: unknown
  width?: number
  height?: number
}

export interface ClipEdge {
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

export interface PastedNode {
  id: string
  type: string
  data: Record<string, unknown>
  position: { x: number; y: number }
  style?: unknown
  width?: number
  height?: number
}

export interface PastedEdge {
  id: string
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

/**
 * @param clip   剪贴板内容(节点保留原 id 以便内部边重连)
 * @param offset 整体落位偏移(连续粘贴递增,避免叠在同一处)
 * @param makeId 发号器(注入便于测试确定性发号)
 */
export function buildPastedGraph(
  clip: { nodes: ClipNode[]; edges: ClipEdge[] },
  offset: number,
  makeId: () => string,
): { nodes: PastedNode[]; edges: PastedEdge[] } {
  const idMap = new Map<string, string>()
  const nodes: PastedNode[] = clip.nodes.map((cn) => {
    const id = makeId()
    idMap.set(cn.id, id)
    const node: PastedNode = {
      id,
      type: cn.type,
      data: structuredClone(cn.data),
      position: { x: cn.position.x + offset, y: cn.position.y + offset },
      style: cn.style ?? { width: 320 },
    }
    if (cn.width != null) node.width = cn.width
    if (cn.height != null) node.height = cn.height
    return node
  })
  // 内部边:两端都在选区内(复制时已过滤),按 id 映射重连;映射缺失则丢弃(防御)。
  const edges: PastedEdge[] = clip.edges
    .map((e) => {
      const source = idMap.get(e.source)
      const target = idMap.get(e.target)
      if (!source || !target) return null
      return { id: makeId(), source, sourceHandle: e.sourceHandle, target, targetHandle: e.targetHandle }
    })
    .filter((e): e is PastedEdge => e !== null)
  return { nodes, edges }
}
