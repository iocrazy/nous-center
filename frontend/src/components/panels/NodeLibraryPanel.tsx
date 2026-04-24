import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import FloatingPanel from '../layout/FloatingPanel'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { PLUGIN_CATEGORIES } from '../../models/nodeRegistry'

// m09 v3: 5 个固定分组对齐 mockup（输入 / AI / 逻辑 / 音频 / 输出）。
// 旧版按"who 注册了它"切（io / tts / declarative），不符合 user mental
// model。现在按"做什么"切，跟 mockup 1:1。

interface NodeCategory {
  name: string
  label: string
  color: string
  nodes: { type: NodeType; dotColor: string }[]
}

const BUILTIN_CATEGORIES: NodeCategory[] = [
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
    name: 'output',
    label: '输出',
    color: 'var(--info)',
    nodes: [
      { type: 'text_output', dotColor: 'var(--info)' },
      { type: 'output', dotColor: 'var(--info)' },
    ],
  },
]

export default function NodeLibraryPanel() {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/reactflow', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  // 插件包仍然按它们自己的 category 渲染在下面（保留可发现性）。
  const allCategories: NodeCategory[] = [
    ...BUILTIN_CATEGORIES,
    ...PLUGIN_CATEGORIES.map((c) => ({
      name: c.name,
      label: c.label || c.name,
      color: c.color,
      nodes: c.nodes,
    })),
  ]

  return (
    <FloatingPanel
      title="节点库"
      searchPlaceholder="搜索节点..."
      onSearch={setSearch}
    >
      {allCategories.map((cat) => {
        const filteredNodes = cat.nodes.filter((n) => {
          const def = NODE_DEFS[n.type]
          if (!def) return false
          return !search || def.label.toLowerCase().includes(search.toLowerCase())
        })
        if (filteredNodes.length === 0 && search) return null
        const isCollapsed = collapsed[cat.name]

        return (
          <div key={cat.name}>
            <div
              className="flex items-center gap-1.5 rounded cursor-pointer"
              style={{ padding: '5px 8px', fontSize: 12, color: 'var(--text)', transition: 'background 0.1s' }}
              onClick={() => setCollapsed({ ...collapsed, [cat.name]: !isCollapsed })}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-hover)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
            >
              {isCollapsed ? <ChevronRight size={10} color="var(--muted)" /> : <ChevronDown size={10} color="var(--muted)" />}
              <span style={{ color: cat.color }}>{cat.label}</span>
              <span
                className="ml-auto"
                style={{
                  fontSize: 10,
                  color: 'var(--muted-strong)',
                  background: 'var(--bg-hover)',
                  padding: '0 5px',
                  borderRadius: 8,
                }}
              >
                {cat.nodes.length}
              </span>
            </div>
            {!isCollapsed &&
              filteredNodes.map(({ type, dotColor }) => (
                <div
                  key={type}
                  className="flex items-center gap-1.5 rounded select-none"
                  style={{
                    padding: '4px 8px 4px 28px',
                    fontSize: 11,
                    color: 'var(--muted)',
                    cursor: 'grab',
                    transition: 'all 0.1s',
                  }}
                  draggable
                  onDragStart={(e) => onDragStart(e, type)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--bg-hover)'
                    e.currentTarget.style.color = 'var(--text)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent'
                    e.currentTarget.style.color = 'var(--muted)'
                  }}
                >
                  <span
                    className="shrink-0 rounded-full"
                    style={{ width: 6, height: 6, background: dotColor }}
                  />
                  {NODE_DEFS[type]?.label ?? type}
                </div>
              ))}
          </div>
        )
      })}
    </FloatingPanel>
  )
}
