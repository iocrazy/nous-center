import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import FloatingPanel from '../layout/FloatingPanel'
import { NODE_DEFS } from '../../models/workflow'
import { getMergedCategories } from '../nodes/nodeChoices'

// m09 v3: 固定分组对齐 mockup（输入 / AI / 逻辑 / 音频 / 图像 / 输出），plugin
// 节点按 category merge 进同名组。分类与端口口径统一抽到 ../nodes/nodeChoices.ts,
// 与「端口拖出建节点」「右键建节点」菜单共用,避免分叉。

export default function NodeLibraryPanel() {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/reactflow', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  const allCategories = getMergedCategories()

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
