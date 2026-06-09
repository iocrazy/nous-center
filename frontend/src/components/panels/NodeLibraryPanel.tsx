import { useState } from 'react'
import {
  ChevronDown, ChevronRight, Box,
  FileInput, FileOutput, Sparkles, GitBranch, AudioLines, Image as ImageIcon, Mic,
  type LucideIcon,
} from 'lucide-react'
import FloatingPanel from '../layout/FloatingPanel'
import { NODE_DEFS } from '../../models/workflow'
import { getMergedCategories } from '../nodes/nodeChoices'
import { PORT_TYPE_COLORS } from '../nodes/portColors'

// 类目图标(按 category name;plugin/未知类目回退 Box)。
const CAT_ICON: Record<string, LucideIcon> = {
  input: FileInput,
  ai: Sparkles,
  logic: GitBranch,
  audio: AudioLines,
  image: ImageIcon,
  output: FileOutput,
  tts: Mic,
}

// 节点端口类型摘要(去重)。
function portTypes(type: string): { ins: string[]; outs: string[] } {
  const def = NODE_DEFS[type]
  if (!def) return { ins: [], outs: [] }
  const uniq = (a: { type: string }[]) => [...new Set(a.map((p) => p.type))]
  return { ins: uniq(def.inputs), outs: uniq(def.outputs) }
}

// 端口类型色点行(最多 4 个)。
function PortDots({ types }: { types: string[] }) {
  return (
    <span className="flex items-center gap-0.5">
      {types.slice(0, 4).map((t, i) => (
        <span
          key={i}
          className="shrink-0 rounded-full"
          style={{ width: 5, height: 5, background: PORT_TYPE_COLORS[t] ?? 'var(--muted-strong)' }}
        />
      ))}
    </span>
  )
}

// m09 v3: 固定分组对齐 mockup（输入 / AI / 逻辑 / 音频 / 图像 / 输出），plugin
// 节点按 category merge 进同名组。分类与端口口径统一抽到 ../nodes/nodeChoices.ts,
// 与「端口拖出建节点」「右键建节点」菜单共用,避免分叉。

// 命中高亮:把 label 按查询大小写无关切分,命中段高亮。
function highlight(label: string, q: string) {
  if (!q) return label
  const i = label.toLowerCase().indexOf(q.toLowerCase())
  if (i < 0) return label
  return (
    <>
      {label.slice(0, i)}
      <span style={{ color: 'var(--accent)', fontWeight: 700 }}>{label.slice(i, i + q.length)}</span>
      {label.slice(i + q.length)}
    </>
  )
}

export default function NodeLibraryPanel() {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/reactflow', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  const allCategories = getMergedCategories()
  const q = search.trim().toLowerCase()
  const searching = q.length > 0

  // 搜索匹配:节点 label / 节点类型 id / 所属类目名(任一含 q)。
  const matchNode = (type: string, catLabel: string) => {
    const def = NODE_DEFS[type]
    if (!def) return false
    if (!q) return true
    return (
      def.label.toLowerCase().includes(q) ||
      type.toLowerCase().includes(q) ||
      catLabel.toLowerCase().includes(q)
    )
  }

  // 先派生每类匹配结果(纯计算,不在 render 中可变累加),再求总数。
  const cats = allCategories.map((cat) => ({
    cat,
    filteredNodes: cat.nodes.filter((n) => matchNode(n.type, cat.label)),
  }))
  const totalMatched = cats.reduce((s, c) => s + c.filteredNodes.length, 0)

  const body = cats.map(({ cat, filteredNodes }) => {
    if (filteredNodes.length === 0 && searching) return null
    // 搜索时强制展开(修:折叠分类下命中节点搜不到的 bug)。
    const isCollapsed = !searching && collapsed[cat.name]

    return (
      <div key={cat.name}>
        <div
          className="flex items-center gap-1.5 rounded cursor-pointer"
          style={{ padding: '5px 8px', fontSize: 12, color: 'var(--text)', transition: 'background 0.1s' }}
          onClick={() => setCollapsed({ ...collapsed, [cat.name]: !collapsed[cat.name] })}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-hover)' }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
        >
          {isCollapsed ? <ChevronRight size={10} color="var(--muted)" /> : <ChevronDown size={10} color="var(--muted)" />}
          {(() => { const Icon = CAT_ICON[cat.name] ?? Box; return <Icon size={12} color={cat.color} /> })()}
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
            {/* 搜索时显示 匹配数/总数,平时显示总数。 */}
            {searching ? `${filteredNodes.length}/${cat.nodes.length}` : cat.nodes.length}
          </span>
        </div>
        {!isCollapsed &&
          filteredNodes.map(({ type, dotColor }) => {
            const { ins, outs } = portTypes(type)
            const title = `输入: ${ins.join(' / ') || '无'}  →  输出: ${outs.join(' / ') || '无'}`
            const ioDots = outs.length ? outs : ins
            return (
            <div
              key={type}
              className="flex items-center gap-1.5 rounded select-none"
              title={title}
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
              <span className="truncate">{highlight(NODE_DEFS[type]?.label ?? type, q)}</span>
              {/* 右侧:节点输出(无输出则输入)端口类型色点提示 */}
              {ioDots.length > 0 && <span className="ml-auto pl-1"><PortDots types={ioDots} /></span>}
            </div>
            )
          })}
      </div>
    )
  })

  return (
    <FloatingPanel
      title="节点库"
      searchPlaceholder="搜索节点..."
      onSearch={setSearch}
    >
      {body}
      {searching && totalMatched === 0 && (
        <div style={{ padding: '16px 8px', fontSize: 11, color: 'var(--muted)', textAlign: 'center' }}>
          无匹配节点
        </div>
      )}
    </FloatingPanel>
  )
}
