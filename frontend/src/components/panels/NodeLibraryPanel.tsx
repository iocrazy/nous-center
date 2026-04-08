import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import FloatingPanel from '../layout/FloatingPanel'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { NODE_CATEGORIES, PLUGIN_CATEGORIES } from '../../models/nodeRegistry'

interface NodeCategory {
  name: string
  color: string
  nodes: { type: NodeType; dotColor: string }[]
}

const BUILTIN_CATEGORIES: NodeCategory[] = [
  ...NODE_CATEGORIES.map((c) => ({
    name: c.name,
    color: c.color,
    nodes: c.nodes,
  })),
  {
    name: 'io',
    color: 'var(--info)',
    nodes: [
      { type: 'text_input', dotColor: 'var(--ok)' },
      { type: 'text_output', dotColor: 'var(--info)' },
    ],
  },
  {
    name: 'tts',
    color: 'var(--accent)',
    nodes: [
      { type: 'ref_audio', dotColor: 'var(--accent-2)' },
      { type: 'tts_engine', dotColor: 'var(--accent)' },
      { type: 'output', dotColor: 'var(--info)' },
    ],
  },
  {
    name: 'audio_processing',
    color: 'var(--info)',
    nodes: [
      { type: 'resample', dotColor: 'var(--info)' },
      { type: 'concat', dotColor: 'var(--info)' },
      { type: 'mixer', dotColor: 'var(--info)' },
      { type: 'bgm_mix', dotColor: 'var(--purple)' },
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

  return (
    <FloatingPanel
      title="Node Library"
      searchPlaceholder="Search Nodes..."
      onSearch={setSearch}
    >
      {[...BUILTIN_CATEGORIES, ...PLUGIN_CATEGORIES].map((cat) => {
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
              <span style={{ color: cat.color }}>{cat.name}</span>
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
