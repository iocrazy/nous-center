import { useState, useRef, useEffect, useCallback } from 'react'
import { Search, X, RefreshCw, Radio } from 'lucide-react'
import { usePanelStore } from '../../stores/panel'
import {
  useRequestLogs,
  useAppLogs,
  useFrontendLogs,
  useAuditLogs,
  type LogItem,
} from '../../api/logs'

// ── Types ──────────────────────────────────────────────────────────────────

type TabId = 'requests' | 'app' | 'frontend' | 'audit'

interface TimeRange {
  label: string
  value: string
  ms: number
}

const TIME_RANGES: TimeRange[] = [
  { label: '15m', value: '15m', ms: 15 * 60 * 1000 },
  { label: '1h', value: '1h', ms: 60 * 60 * 1000 },
  { label: '24h', value: '24h', ms: 24 * 60 * 60 * 1000 },
  { label: '3d', value: '3d', ms: 3 * 24 * 60 * 60 * 1000 },
  { label: '7d', value: '7d', ms: 7 * 24 * 60 * 60 * 1000 },
]

// ── Helpers ────────────────────────────────────────────────────────────────

function sinceFromRange(rangeMs: number): string {
  const d = new Date(Date.now() - rangeMs)
  return d.toISOString()
}

function formatTs(ts: string): string {
  try {
    return new Date(ts).toLocaleString(undefined, {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  } catch {
    return ts
  }
}

// ── Badge components ───────────────────────────────────────────────────────

function MethodBadge({ method }: { method: string }) {
  const m = (method ?? '').toUpperCase()
  const color =
    m === 'GET' ? 'var(--ok)' :
    m === 'POST' ? 'var(--accent)' :
    m === 'PUT' ? 'var(--warning, #f59e0b)' :
    m === 'DELETE' ? 'var(--error)' :
    'var(--muted)'

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.04em',
        background: `color-mix(in srgb, ${color} 18%, transparent)`,
        color,
        border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
      }}
    >
      {m || '—'}
    </span>
  )
}

function StatusBadge({ status }: { status: number | string | null | undefined }) {
  const s = Number(status)
  const color =
    s >= 500 ? 'var(--error)' :
    s >= 400 ? 'var(--warning, #f59e0b)' :
    s >= 200 ? 'var(--ok)' :
    'var(--muted)'

  return (
    <span style={{ color, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
      {status ?? '—'}
    </span>
  )
}

function LevelBadge({ level }: { level: string | null | undefined }) {
  const l = (level ?? '').toUpperCase()
  const color =
    l === 'ERROR' || l === 'CRITICAL' ? 'var(--error)' :
    l === 'WARNING' || l === 'WARN' ? 'var(--warning, #f59e0b)' :
    l === 'DEBUG' ? 'var(--muted)' :
    'var(--info, #06b6d4)'

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 700,
        background: `color-mix(in srgb, ${color} 18%, transparent)`,
        color,
        border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
      }}
    >
      {l || 'INFO'}
    </span>
  )
}

// ── Table components ───────────────────────────────────────────────────────

function Th({ children, w }: { children: React.ReactNode; w?: string | number }) {
  return (
    <th
      style={{
        padding: '6px 10px',
        textAlign: 'left',
        fontSize: 11,
        fontWeight: 600,
        color: 'var(--muted)',
        borderBottom: '1px solid var(--border)',
        whiteSpace: 'nowrap',
        width: w,
        background: 'var(--bg-accent)',
        position: 'sticky',
        top: 0,
        zIndex: 1,
      }}
    >
      {children}
    </th>
  )
}

function Td({ children, mono }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <td
      style={{
        padding: '5px 10px',
        fontSize: 12,
        color: 'var(--text)',
        borderBottom: '1px solid var(--border)',
        fontFamily: mono ? 'var(--font-mono, monospace)' : undefined,
        maxWidth: 280,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </td>
  )
}

function EmptyRow({ cols }: { cols: number }) {
  return (
    <tr>
      <td
        colSpan={cols}
        style={{
          padding: '40px 0',
          textAlign: 'center',
          color: 'var(--muted)',
          fontSize: 13,
        }}
      >
        No entries found
      </td>
    </tr>
  )
}

function LoadingRow({ cols }: { cols: number }) {
  return (
    <tr>
      <td
        colSpan={cols}
        style={{
          padding: '40px 0',
          textAlign: 'center',
          color: 'var(--muted)',
          fontSize: 13,
        }}
      >
        Loading…
      </td>
    </tr>
  )
}

// ── Tab tables ─────────────────────────────────────────────────────────────

function RequestTable({ items, loading }: { items: LogItem[]; loading: boolean }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <Th w={140}>Time</Th>
          <Th w={70}>Method</Th>
          <Th>Path</Th>
          <Th w={60}>Status</Th>
          <Th w={80}>Duration</Th>
          <Th w={60}>IP</Th>
        </tr>
      </thead>
      <tbody>
        {loading && items.length === 0 ? (
          <LoadingRow cols={6} />
        ) : items.length === 0 ? (
          <EmptyRow cols={6} />
        ) : (
          items.map((row) => (
            <tr key={row.id} className="log-row">
              <Td mono>{formatTs(row.timestamp)}</Td>
              <Td><MethodBadge method={String(row.method ?? '')} /></Td>
              <Td mono>{String(row.path ?? '—')}</Td>
              <Td><StatusBadge status={row.status as number} /></Td>
              <Td>{row.duration_ms != null ? `${Number(row.duration_ms)}ms` : '—'}</Td>
              <Td>{String(row.ip ?? '—')}</Td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  )
}

function AppTable({ items, loading }: { items: LogItem[]; loading: boolean }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <Th w={140}>Time</Th>
          <Th w={80}>Level</Th>
          <Th w={120}>Module</Th>
          <Th>Message</Th>
        </tr>
      </thead>
      <tbody>
        {loading && items.length === 0 ? (
          <LoadingRow cols={4} />
        ) : items.length === 0 ? (
          <EmptyRow cols={4} />
        ) : (
          items.map((row) => (
            <tr key={row.id} className="log-row">
              <Td mono>{formatTs(row.timestamp)}</Td>
              <Td><LevelBadge level={String(row.level ?? '')} /></Td>
              <Td mono>{String(row.module ?? '—')}</Td>
              <Td>{String(row.message ?? '—')}</Td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  )
}

function FrontendTable({ items, loading }: { items: LogItem[]; loading: boolean }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <Th w={140}>Time</Th>
          <Th w={80}>Type</Th>
          <Th>Message</Th>
          <Th w={120}>Page</Th>
        </tr>
      </thead>
      <tbody>
        {loading && items.length === 0 ? (
          <LoadingRow cols={4} />
        ) : items.length === 0 ? (
          <EmptyRow cols={4} />
        ) : (
          items.map((row) => (
            <tr key={row.id} className="log-row">
              <Td mono>{formatTs(row.timestamp)}</Td>
              <Td><LevelBadge level={String(row.type ?? '')} /></Td>
              <Td>{String(row.message ?? '—')}</Td>
              <Td mono>{String(row.page ?? '—')}</Td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  )
}

function AuditTable({ items, loading }: { items: LogItem[]; loading: boolean }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <Th w={140}>Time</Th>
          <Th w={80}>Action</Th>
          <Th w={120}>Path</Th>
          <Th w={80}>Method</Th>
          <Th w={100}>IP</Th>
          <Th>Detail</Th>
        </tr>
      </thead>
      <tbody>
        {loading && items.length === 0 ? (
          <LoadingRow cols={6} />
        ) : items.length === 0 ? (
          <EmptyRow cols={6} />
        ) : (
          items.map((row) => (
            <tr key={row.id} className="log-row">
              <Td mono>{formatTs(row.timestamp)}</Td>
              <Td><LevelBadge level={String(row.action ?? '')} /></Td>
              <Td mono>{String(row.path ?? '—')}</Td>
              <Td><MethodBadge method={String(row.method ?? '')} /></Td>
              <Td>{String(row.ip ?? '—')}</Td>
              <Td>{String(row.detail ?? '—')}</Td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  )
}

// ── Main overlay ───────────────────────────────────────────────────────────

const TABS: { id: TabId; label: string }[] = [
  { id: 'requests', label: 'Request Logs' },
  { id: 'app', label: 'App Logs' },
  { id: 'frontend', label: 'Frontend Logs' },
  { id: 'audit', label: 'Audit Logs' },
]

export default function LogsOverlay() {
  const setOverlay = usePanelStore((s) => s.setOverlay)

  const [activeTab, setActiveTab] = useState<TabId>('requests')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [timeRange, setTimeRange] = useState<TimeRange>(TIME_RANGES[1]) // default 1h
  const [live, setLive] = useState(true)

  // Debounce search input
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const handleSearch = useCallback((val: string) => {
    setSearch(val)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setDebouncedSearch(val), 300)
  }, [])

  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [])

  const queryParams = {
    search: debouncedSearch || undefined,
    since: sinceFromRange(timeRange.ms),
    limit: 200,
  }

  const reqQuery = useRequestLogs(queryParams, activeTab === 'requests' && live)
  const appQuery = useAppLogs(queryParams, activeTab === 'app' && live)
  const feQuery = useFrontendLogs(queryParams, activeTab === 'frontend' && live)
  const auditQuery = useAuditLogs(queryParams, activeTab === 'audit' && live)

  const currentQuery =
    activeTab === 'requests' ? reqQuery :
    activeTab === 'app' ? appQuery :
    activeTab === 'frontend' ? feQuery :
    auditQuery

  const items = currentQuery.data?.items ?? []
  const total = currentQuery.data?.total ?? 0
  const loading = currentQuery.isLoading || currentQuery.isFetching

  const handleRefresh = () => {
    currentQuery.refetch()
  }

  return (
    <div
      className="absolute inset-0 z-30 flex flex-col"
      style={{ background: 'var(--bg)', overflow: 'hidden' }}
    >
      {/* Header */}
      <div
        style={{
          padding: '12px 16px 0',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-accent)',
          flexShrink: 0,
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <h2 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-strong)', margin: 0 }}>
            Logs
          </h2>
          <div className="flex items-center gap-2">
            {/* Total count */}
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>
              {total.toLocaleString()} entries
            </span>

            {/* Refresh button */}
            <button
              onClick={handleRefresh}
              title="Refresh"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 28,
                height: 28,
                borderRadius: 5,
                border: '1px solid var(--border)',
                background: 'transparent',
                color: loading ? 'var(--accent)' : 'var(--muted)',
                cursor: 'pointer',
                transition: 'color 0.12s',
              }}
            >
              <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            </button>

            {/* Live toggle */}
            <button
              onClick={() => setLive((v) => !v)}
              title={live ? 'Pause live updates' : 'Enable live updates'}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '3px 8px',
                borderRadius: 5,
                border: `1px solid ${live ? 'var(--ok)' : 'var(--border)'}`,
                background: live ? 'color-mix(in srgb, var(--ok) 15%, transparent)' : 'transparent',
                color: live ? 'var(--ok)' : 'var(--muted)',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: 600,
                transition: 'all 0.12s',
              }}
            >
              <Radio size={11} />
              Live
            </button>

            {/* Close */}
            <button
              onClick={() => setOverlay(null)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 28,
                height: 28,
                borderRadius: 5,
                border: '1px solid var(--border)',
                background: 'transparent',
                color: 'var(--muted)',
                cursor: 'pointer',
              }}
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* Toolbar: search + time range */}
        <div className="flex items-center gap-2 mb-3">
          {/* Search */}
          <div className="relative flex-1" style={{ maxWidth: 340 }}>
            <Search
              size={13}
              style={{
                position: 'absolute',
                left: 9,
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--muted)',
                pointerEvents: 'none',
              }}
            />
            <input
              value={search}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="Search logs…"
              style={{
                width: '100%',
                paddingLeft: 28,
                paddingRight: 8,
                height: 30,
                borderRadius: 5,
                border: '1px solid var(--border)',
                background: 'var(--bg)',
                color: 'var(--text)',
                fontSize: 12,
                outline: 'none',
              }}
            />
          </div>

          {/* Time range buttons */}
          <div className="flex items-center gap-1">
            {TIME_RANGES.map((tr) => (
              <button
                key={tr.value}
                onClick={() => setTimeRange(tr)}
                style={{
                  padding: '3px 8px',
                  borderRadius: 5,
                  border: `1px solid ${timeRange.value === tr.value ? 'var(--accent)' : 'var(--border)'}`,
                  background: timeRange.value === tr.value ? 'var(--accent-subtle)' : 'transparent',
                  color: timeRange.value === tr.value ? 'var(--accent)' : 'var(--muted)',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.12s',
                }}
              >
                {tr.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '6px 14px',
                fontSize: 12,
                fontWeight: activeTab === tab.id ? 600 : 400,
                color: activeTab === tab.id ? 'var(--accent)' : 'var(--muted)',
                border: 'none',
                borderBottom: `2px solid ${activeTab === tab.id ? 'var(--accent)' : 'transparent'}`,
                background: 'transparent',
                cursor: 'pointer',
                transition: 'all 0.12s',
                marginBottom: -1,
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Table body */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        <style>{`
          .log-row:hover td { background: var(--bg-hover); }
          @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        `}</style>

        {activeTab === 'requests' && <RequestTable items={items} loading={loading} />}
        {activeTab === 'app' && <AppTable items={items} loading={loading} />}
        {activeTab === 'frontend' && <FrontendTable items={items} loading={loading} />}
        {activeTab === 'audit' && <AuditTable items={items} loading={loading} />}
      </div>
    </div>
  )
}
