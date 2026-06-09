// 分组框几何 + 成员计数(PR-4 自适应 / 计数副标题)。纯函数,便于单测。
import { NODE_DEFS, type NodeType } from '../../models/workflow'

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export const GROUP_PAD = 24
export const GROUP_HEADER = 30

/**
 * 按成员实测矩形算分组包围盒 + padding + 顶部留头(对齐 Infinite-Canvas 自适应)。
 * 空成员返回 null(交由调用方回退)。
 */
export function computeGroupBounds(
  members: Rect[],
  pad: number = GROUP_PAD,
  header: number = GROUP_HEADER,
): Rect | null {
  if (!members.length) return null
  const minX = Math.min(...members.map((m) => m.x))
  const minY = Math.min(...members.map((m) => m.y))
  const maxX = Math.max(...members.map((m) => m.x + m.width))
  const maxY = Math.max(...members.map((m) => m.y + m.height))
  return {
    x: minX - pad,
    y: minY - pad - header,
    width: maxX - minX + pad * 2,
    height: maxY - minY + pad * 2 + header,
  }
}

export type NodeCategory = 'image' | 'prompt' | 'other'

/** 节点类型 → 计数类目:有 image 端口 = 图片;text_input = 提示词;其余 other。 */
export function classifyNodeType(type: string): NodeCategory {
  if (type === 'text_input') return 'prompt'
  const def = NODE_DEFS[type as NodeType]
  if (def) {
    const ports = [...def.inputs, ...def.outputs]
    if (ports.some((p) => p.type === 'image')) return 'image'
  }
  return 'other'
}

export interface GroupCounts {
  images: number
  prompts: number
  others: number
}

export function countGroupMembers(types: string[]): GroupCounts {
  const counts: GroupCounts = { images: 0, prompts: 0, others: 0 }
  for (const t of types) {
    const c = classifyNodeType(t)
    if (c === 'image') counts.images += 1
    else if (c === 'prompt') counts.prompts += 1
    else counts.others += 1
  }
  return counts
}

/** 副标题:「N张图片 · M个提示词 · K个节点 已成组」(零项省略),对齐截图。 */
export function groupSubtitle(counts: GroupCounts): string {
  const parts: string[] = []
  if (counts.images) parts.push(`${counts.images}张图片`)
  if (counts.prompts) parts.push(`${counts.prompts}个提示词`)
  if (counts.others) parts.push(`${counts.others}个节点`)
  const head = parts.length ? parts.join(' · ') : '空'
  return `${head} 已成组`
}
