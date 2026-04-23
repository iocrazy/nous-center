import { useMemo, useState } from 'react'
import { Activity, AppWindow, Cpu, Image as ImageIcon, Plus, Search } from 'lucide-react'
import {
  endpointFor,
  useServices,
  type ServiceCategory,
  type ServiceRow,
} from '../api/services'
import CreateServiceDialog from '../components/services/CreateServiceDialog'

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
    else window.history.pushState({}, '', `/services/${id}`)
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
            <button
              type="button"
              onClick={() => setCreateOpen(true)}
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
              快速开通
            </button>
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
              <ServiceCard key={svc.id} svc={svc} onOpen={() => goDetail(svc.id)} />
            ))}
          </div>
        )}

        <CreateServiceDialog
          open={createOpen}
          onClose={() => setCreateOpen(false)}
          onCreated={(id) => goDetail(id)}
        />
      </div>
    </div>
  )
}

function ServiceCard({ svc, onOpen }: { svc: ServiceRow; onOpen: () => void }) {
  const statusStyle = STATUS_STYLES[svc.status] ?? STATUS_STYLES.active
  const inactive = svc.status !== 'active'
  return (
    <button
      type="button"
      onClick={onOpen}
      style={{
        textAlign: 'left',
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderLeft: inactive ? '1px solid var(--border)' : '3px solid var(--accent-2, #22c55e)',
        borderRadius: 8,
        padding: 14,
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        opacity: inactive ? 0.7 : 1,
        color: 'var(--text)',
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
        <Tag>{(svc.category ?? 'app').toUpperCase()}</Tag>
        <Tag>{svc.source_type === 'workflow' ? '来自 Workflow' : '快速开通'}</Tag>
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
