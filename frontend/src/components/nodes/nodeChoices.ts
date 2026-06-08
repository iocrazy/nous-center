/**
 * nodeChoices — 画布「建节点」的统一数据源。把节点分类(内置 BUILTIN_CATEGORIES +
 * 运行时 PLUGIN_CATEGORIES)合并成一份列表,供:
 *   1. 左侧节点库面板(NodeLibraryPanel)
 *   2. 端口拖到空白 → 快捷建相连节点菜单(借鉴 Infinite-Canvas)
 *   3. 画布右键 → 快捷建节点菜单
 * 三处共用,避免分类/端口口径分叉。
 */
import { NODE_DEFS, type NodeType, type PortType } from '../../models/workflow'
import { PLUGIN_CATEGORIES } from '../../models/nodeRegistry'

export interface NodeCategory {
  name: string
  label: string
  color: string
  nodes: { type: NodeType; dotColor: string }[]
}

// m09 v3:固定分组按「做什么」切(输入/AI/逻辑/音频/图像/输出),与 mockup 1:1。
// plugin 节点按 category merge 进同名内置组(见 getMergedCategories)。
export const BUILTIN_CATEGORIES: NodeCategory[] = [
  {
    name: 'input',
    label: '输入',
    color: 'var(--ok)',
    nodes: [
      { type: 'text_input', dotColor: 'var(--ok)' },
      { type: 'multimodal_input', dotColor: 'var(--purple)' },
      { type: 'ref_audio', dotColor: 'var(--accent-2)' },
    ],
  },
  {
    name: 'ai',
    label: 'AI 节点',
    color: 'var(--purple)',
    nodes: [
      { type: 'llm', dotColor: 'var(--purple)' },
      { type: 'prompt_template', dotColor: 'var(--purple)' },
      { type: 'agent', dotColor: 'var(--purple)' },
    ],
  },
  {
    name: 'logic',
    label: '逻辑',
    color: 'var(--accent)',
    nodes: [
      { type: 'if_else', dotColor: 'var(--accent)' },
      { type: 'python_exec', dotColor: 'var(--accent-2)' },
    ],
  },
  {
    name: 'audio',
    label: '音频处理',
    color: 'var(--info)',
    nodes: [
      { type: 'tts_engine', dotColor: 'var(--accent)' },
      { type: 'resample', dotColor: 'var(--info)' },
      { type: 'concat', dotColor: 'var(--info)' },
      { type: 'mixer', dotColor: 'var(--info)' },
      { type: 'bgm_mix', dotColor: 'var(--purple)' },
    ],
  },
  {
    name: 'image',
    label: '图像',
    color: 'var(--info)',
    nodes: [{ type: 'image_output', dotColor: 'var(--info)' }],
  },
  {
    name: 'output',
    label: '输出',
    color: 'var(--info)',
    nodes: [
      { type: 'text_output', dotColor: 'var(--info)' },
      { type: 'output', dotColor: 'var(--info)' },
    ],
  },
]

/**
 * 合并内置 + plugin 分类(plugin 按 category 名 merge 进同名内置组,否则独立成组)。
 * 每次调用返回新对象(不可变;不污染 BUILTIN_CATEGORIES)。
 */
export function getMergedCategories(): NodeCategory[] {
  const merged: NodeCategory[] = BUILTIN_CATEGORIES.map((c) => ({ ...c, nodes: [...c.nodes] }))
  const byName: Record<string, NodeCategory> = {}
  for (const c of merged) byName[c.name] = c

  const standalone: NodeCategory[] = []
  for (const c of PLUGIN_CATEGORIES) {
    const rawName = c.name.startsWith('plugin:') ? c.name.slice('plugin:'.length) : c.name
    const target = byName[rawName]
    if (target) {
      const existing = new Set(target.nodes.map((n) => n.type))
      for (const n of c.nodes) if (!existing.has(n.type)) target.nodes.push(n)
    } else {
      standalone.push({ name: c.name, label: c.label || c.name, color: c.color, nodes: c.nodes })
    }
  }
  return [...merged, ...standalone]
}

export interface NodeChoice {
  type: NodeType
  label: string
  color: string
}

/** 合并分类拍平成 {type,label,color}(去重)。 */
export function getAllChoices(): NodeChoice[] {
  const out: NodeChoice[] = []
  const seen = new Set<string>()
  for (const cat of getMergedCategories()) {
    for (const n of cat.nodes) {
      if (seen.has(n.type)) continue
      seen.add(n.type)
      out.push({ type: n.type, label: NODE_DEFS[n.type]?.label ?? n.type, color: n.dotColor })
    }
  }
  return out
}

/**
 * 从一个**输出**端口(type=portType)拖出 → 能接收它的候选节点(有同类型**输入**口)。
 * 匹配规则与 isValidConnection 一致(严格同类型)。
 */
export function choicesAcceptingInput(portType: PortType): NodeChoice[] {
  return getAllChoices().filter((c) =>
    (NODE_DEFS[c.type]?.inputs ?? []).some((p) => p.type === portType),
  )
}

/** 从一个**输入**端口(type=portType)拖出 → 能供给它的候选节点(有同类型**输出**口)。 */
export function choicesProvidingOutput(portType: PortType): NodeChoice[] {
  return getAllChoices().filter((c) =>
    (NODE_DEFS[c.type]?.outputs ?? []).some((p) => p.type === portType),
  )
}

/** 节点 type 上第一个匹配 portType 的输入/输出 handle id。 */
export function firstInputHandle(type: NodeType, portType: PortType): string | undefined {
  return (NODE_DEFS[type]?.inputs ?? []).find((p) => p.type === portType)?.id
}
export function firstOutputHandle(type: NodeType, portType: PortType): string | undefined {
  return (NODE_DEFS[type]?.outputs ?? []).find((p) => p.type === portType)?.id
}
