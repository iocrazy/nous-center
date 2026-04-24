import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  AppWindow,
  ChevronDown,
  Cpu,
  Image as ImageIcon,
  Plus,
  Search,
} from 'lucide-react'
import {
  endpointFor,
  useServices,
  type ServiceCategory,
  type ServiceRow,
} from '../api/services'
import CreateServiceDialog from '../components/services/CreateServiceDialog'
import { useToastStore } from '../stores/toast'

type FilterTab = 'all' | ServiceCategory

const TAB_DEFS: { id: FilterTab; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'llm', label: 'LLM' },
  { id: 'tts', label: 'TTS' },
  { id: 'vl', label: 'VL' },
  { id: 'app', label: '其他' },
]

export interface ServicesListProps {
  /** Optional callback when user clicks a row. Defaults to navigating to the
   *  detail page; consumers can override (e.g. open in modal). */
  onOpen?: (id: string) => void
}

export default function ServicesList({ onOpen }: ServicesListProps) {
  const { data: services, isLoading, error } = useServices()
  const navigate = useNavigate()
  const [tab, setTab] = useState<FilterTab>('all')
  const [search, setSearch] = useState('')
  const [createOpen, setCreateOpen] = useState(false)

  const counts = useMemo(() => buildCounts(services ?? []), [services])
  const filtered = useMemo(
    () => filter(services ?? [], tab, search),
    [services, tab, search],
  )

  const goDetail = (id: string) => {
    if (onOpen) onOpen(id)
    else navigate(`/services/${id}`)
  }

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'auto',
        background: 'var(--bg)',
      }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
        {/* header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', marginBottom: 14 }}>
          <div style={{ flex: 1 }}>
            <h1 style={{ fontSize: 20, color: 'var(--text)', fontWeight: 600 }}>服务</h1>
            <p style={{ fontSize: 13, color: 'var(--muted)', marginTop: 4 }}>
              所有对外可调用的服务实例 · 每条服务 = endpoint + schema + 授权 · 通过 API Key 调用
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ position: 'relative' }}>
              <Search
                size={14}
                style={{
                  position: 'absolute',
                  left: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--muted)',
                }}
              />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="搜索服务..."
                style={{
                  width: 220,
                  background: 'var(--bg-accent)',
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  padding: '7px 9px 7px 28px',
                  fontSize: 12,
                }}
              />
            </div>
            <NewServiceMenu
              onQuickProvision={() => setCreateOpen(true)}
              onPublishFromWorkflow={() => navigate('/workflows')}
            />
          </div>
        </div>

        {/* tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 14 }}>
          {TAB_DEFS.map((t) => {
            const n = t.id === 'all' ? services?.length ?? 0 : counts[t.id] ?? 0
            const active = tab === t.id
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                style={{
                  padding: '6px 12px',
                  background: active ? 'var(--accent-subtle, rgba(99,102,241,0.1))' : 'transparent',
                  color: active ? 'var(--accent)' : 'var(--muted)',
                  border: '1px solid',
                  borderColor: active ? 'var(--accent)' : 'var(--border)',
                  borderRadius: 4,
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                {t.label} {n}
              </button>
            )
          })}
        </div>

        {/* body */}
        {isLoading && <Loading />}
        {error && <ErrorBlock message={(error as Error).message} />}
        {services && services.length === 0 && <Empty onCreate={() => setCreateOpen(true)} />}
        {filtered && filtered.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
              gap: 12,
            }}
          >
            {filtered.map((svc) => (
              <ServiceCard
                key={svc.id}
                svc={svc}
                onOpen={() => goDetail(svc.id)}
                onPlayground={() => goDetail(svc.id)}
              />
            ))}
          </div>
        )}

        <FooterHint />

        <CreateServiceDialog
          open={createOpen}
          onClose={() => setCreateOpen(false)}
          onCreated={(id) => goDetail(id)}
        />
      </div>
    </div>
  )
}

// ---------- new service split-button ----------

function NewServiceMenu({
  onQuickProvision,
  onPublishFromWorkflow,
}: {
  onQuickProvision: () => void
  onPublishFromWorkflow: () => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          padding: '7px 12px',
          background: 'var(--accent)',
          color: '#fff',
          border: 'none',
          borderRadius: 4,
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        <Plus size={14} />
        新建服务
        <ChevronDown size={12} style={{ marginLeft: 2 }} />
      </button>
      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            top: 'calc(100% + 4px)',
            right: 0,
            minWidth: 280,
            background: 'var(--bg-elevated, var(--bg))',
            border: '1px solid var(--border)',
            borderRadius: 6,
            boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
            padding: '4px 0',
            zIndex: 30,
          }}
        >
          <MenuItem
            title="快速开通"
            desc="选引擎 · 填参数 · 直接得到服务（适合 LLM/TTS/VL 单步）"
            onClick={() => {
              setOpen(false)
              onQuickProvision()
            }}
          />
          <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />
          <MenuItem
            title="从 Workflow 发布"
            desc="挑一个 Workflow · 指定输入/输出 · 发布（适合多步流程）"
            onClick={() => {
              setOpen(false)
              onPublishFromWorkflow()
            }}
          />
        </div>
      )}
    </div>
  )
}

function MenuItem({
  title,
  desc,
  onClick,
}: {
  title: string
  desc: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        gap: 2,
        padding: '10px 14px',
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        width: '100%',
        textAlign: 'left',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover, var(--bg-accent))')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      <span style={{ fontSize: 12, color: 'var(--text)', fontWeight: 500 }}>{title}</span>
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>{desc}</span>
    </button>
  )
}

// ---------- card ----------

function ServiceCard({
  svc,
  onOpen,
  onPlayground,
}: {
  svc: ServiceRow
  onOpen: () => void
  onPlayground: () => void
}) {
  const statusStyle = STATUS_STYLES[svc.status] ?? STATUS_STYLES.active
  const inactive = svc.status !== 'active'
  const addToast = useToastStore((s) => s.add)

  const copyCurl = (e: React.MouseEvent) => {
    e.stopPropagation()
    const url =
      svc.category === 'llm'
        ? 'https://YOUR_HOST/v1/chat/completions'
        : `https://YOUR_HOST/v1/apps/${svc.name}/run`
    const body =
      svc.category === 'llm'
        ? `{"model": "${svc.name}", "messages": [{"role": "user", "content": "..."}]}`
        : '{}'
    const curl =
      `curl -X POST '${url}' \\\n` +
      `  -H 'Authorization: Bearer YOUR_API_KEY' \\\n` +
      `  -H 'Content-Type: application/json' \\\n` +
      `  -d '${body}'`
    navigator.clipboard
      .writeText(curl)
      .then(() => addToast(`已复制 ${svc.name} 的 curl`, 'info'))
      .catch(() => addToast('复制失败', 'error'))
  }

  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderLeft: inactive ? '1px solid var(--border)' : '3px solid var(--accent-2, #22c55e)',
        borderRadius: 8,
        padding: 14,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        opacity: inactive ? 0.7 : 1,
        color: 'var(--text)',
      }}
    >
      <button
        type="button"
        onClick={onOpen}
        style={{
          textAlign: 'left',
          background: 'transparent',
          border: 'none',
          padding: 0,
          cursor: 'pointer',
          color: 'var(--text)',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
          <CategoryIcon category={svc.category} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 14, color: 'var(--text)', fontWeight: 500 }}>{svc.name}</div>
          </div>
          <span
            style={{
              fontSize: 11,
              padding: '2px 7px',
              borderRadius: 10,
              ...statusStyle,
            }}
          >
            {STATUS_LABEL[svc.status] ?? svc.status}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <CategoryTag category={svc.category} />
          <SourceTag sourceType={svc.source_type} />
          <Tag>v{svc.version}</Tag>
        </div>

        <div
          style={{
            fontFamily: 'var(--mono, monospace)',
            fontSize: 11,
            color: 'var(--muted)',
            padding: '5px 8px',
            background: 'var(--bg)',
            borderRadius: 4,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {endpointFor(svc)}
        </div>
      </button>

      <div
        style={{
          display: 'flex',
          gap: 6,
          paddingTop: 10,
          borderTop: '1px dashed var(--border)',
        }}
      >
        <CardBtn onClick={onOpen}>详情</CardBtn>
        <CardBtn onClick={onPlayground}>Playground</CardBtn>
        <CardBtn onClick={copyCurl}>复制 curl</CardBtn>
      </div>
    </div>
  )
}

function CardBtn({
  onClick,
  children,
}: {
  onClick: (e: React.MouseEvent) => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        flex: 1,
        padding: '5px 0',
        fontSize: 11,
        textAlign: 'center',
        background: 'var(--bg)',
        color: 'var(--muted)',
        border: '1px solid var(--border)',
        borderRadius: 4,
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--text)'
        e.currentTarget.style.borderColor = 'var(--accent)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--muted)'
        e.currentTarget.style.borderColor = 'var(--border)'
      }}
    >
      {children}
    </button>
  )
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontSize: 10,
        padding: '1px 7px',
        borderRadius: 10,
        background: 'var(--bg)',
        color: 'var(--muted)',
      }}
    >
      {children}
    </span>
  )
}

const CATEGORY_TAG_STYLES: Record<string, React.CSSProperties> = {
  llm: { background: 'rgba(34,197,94,0.15)', color: 'var(--accent-2, #22c55e)' },
  tts: { background: 'rgba(168,85,247,0.15)', color: 'var(--purple, #a855f7)' },
  vl: { background: 'rgba(59,130,246,0.15)', color: 'var(--info, #3b82f6)' },
  app: { background: 'var(--bg)', color: 'var(--muted)' },
}

function CategoryTag({ category }: { category: ServiceCategory | null }) {
  const c = (category ?? 'app').toLowerCase()
  const label = (category ?? 'APP').toString().toUpperCase()
  const style = CATEGORY_TAG_STYLES[c] ?? CATEGORY_TAG_STYLES.app
  return (
    <span style={{ fontSize: 10, padding: '1px 7px', borderRadius: 10, ...style }}>{label}</span>
  )
}

function SourceTag({ sourceType }: { sourceType: 'workflow' | 'preset' | 'model' }) {
  const isWorkflow = sourceType === 'workflow'
  return (
    <span
      style={{
        fontSize: 10,
        padding: '1px 7px',
        borderRadius: 10,
        background: isWorkflow
          ? 'var(--accent-subtle, rgba(99,102,241,0.1))'
          : 'var(--bg)',
        color: isWorkflow ? 'var(--accent)' : 'var(--muted)',
        border: isWorkflow ? 'none' : '1px solid var(--border)',
      }}
    >
      {isWorkflow ? '来自 Workflow' : '快速开通'}
    </span>
  )
}

function CategoryIcon({ category }: { category: ServiceCategory | null }) {
  const style = { marginTop: 2, color: 'var(--accent)' } as const
  if (category === 'llm') return <Cpu size={16} style={style} />
  if (category === 'vl') return <ImageIcon size={16} style={style} />
  if (category === 'app') return <AppWindow size={16} style={style} />
  return <Activity size={16} style={style} />
}

const STATUS_LABEL: Record<string, string> = {
  active: '运行中',
  paused: '已暂停',
  deprecated: '已弃用',
  retired: '已下线',
}
const STATUS_STYLES: Record<string, React.CSSProperties> = {
  active: { background: 'rgba(34,197,94,0.15)', color: 'var(--accent-2, #22c55e)' },
  paused: { background: 'rgba(245,158,11,0.15)', color: 'var(--warn, #f59e0b)' },
  deprecated: { background: 'var(--bg)', color: 'var(--muted)', border: '1px solid var(--border)' },
  retired: { background: 'rgba(239,68,68,0.12)', color: 'var(--error, #ef4444)' },
}

function buildCounts(rows: ServiceRow[]): Record<ServiceCategory, number> {
  const out: Record<ServiceCategory, number> = { llm: 0, tts: 0, vl: 0, app: 0 }
  for (const r of rows) if (r.category && out[r.category] !== undefined) out[r.category] += 1
  return out
}

function filter(rows: ServiceRow[], tab: FilterTab, search: string): ServiceRow[] {
  const q = search.trim().toLowerCase()
  return rows.filter((r) => {
    if (tab !== 'all' && r.category !== tab) return false
    if (q && !r.name.toLowerCase().includes(q)) return false
    return true
  })
}

function FooterHint() {
  return (
    <p
      style={{
        fontSize: 11,
        color: 'var(--muted)',
        marginTop: 16,
        padding: '10px 14px',
        background: 'var(--bg-accent)',
        borderRadius: 4,
        borderLeft: '2px solid var(--accent-2, #22c55e)',
        lineHeight: 1.7,
      }}
    >
      <strong style={{ color: 'var(--text)' }}>提示：</strong>
      <br />· <strong>快速开通</strong> 适合简单的单步调用（LLM 直接 chat、TTS 直接合成），系统在后台生成 trivial workflow。
      <br />· <strong>从 Workflow 发布</strong> 适合多步流程（LTX 短剧、图像 pipeline），在编辑器里搭好 DAG、指定输入/输出后发布。
      <br />· 两条路径产出同一种服务对象 — 都有 endpoint + schema + 配额 + API Key 授权。
    </p>
  )
}

function Loading() {
  return (
    <div style={{ textAlign: 'center', padding: 40, color: 'var(--muted)', fontSize: 13 }}>
      加载中…
    </div>
  )
}

function ErrorBlock({ message }: { message: string }) {
  return (
    <div
      style={{
        background: 'rgba(239,68,68,0.1)',
        border: '1px solid var(--error, #ef4444)',
        color: 'var(--error, #ef4444)',
        padding: 14,
        borderRadius: 6,
        fontSize: 13,
      }}
    >
      {message}
    </div>
  )
}

function Empty({ onCreate }: { onCreate: () => void }) {
  return (
    <div
      style={{
        textAlign: 'center',
        padding: 40,
        background: 'var(--bg-accent)',
        border: '1px dashed var(--border)',
        borderRadius: 8,
        color: 'var(--muted)',
      }}
    >
      <div style={{ fontSize: 14, marginBottom: 6 }}>还没有服务</div>
      <div style={{ fontSize: 12, marginBottom: 12 }}>
        从快速开通开始，或在 Workflow 编辑器里发布一个 DAG。
      </div>
      <button
        type="button"
        onClick={onCreate}
        style={{
          padding: '7px 14px',
          background: 'var(--accent)',
          color: '#fff',
          border: 'none',
          borderRadius: 4,
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        快速开通
      </button>
    </div>
  )
}
