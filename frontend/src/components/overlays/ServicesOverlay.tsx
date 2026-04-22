import { useState } from 'react'
import { Activity, AlertTriangle, ChevronRight } from 'lucide-react'
import { useServicesCatalog, type CatalogService } from '../../api/apiGateway'

export default function ServicesOverlay() {
  const { data: services, isLoading, error } = useServicesCatalog()
  const [expandedId, setExpandedId] = useState<number | null>(null)

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
        <div className="mb-5">
          <h1 className="text-[20px] font-semibold" style={{ color: 'var(--fg)' }}>
            服务目录
          </h1>
          <p className="text-[13px] mt-1" style={{ color: 'var(--muted)' }}>
            当前 API Key 可调用的服务与配额余量
          </p>
        </div>

        {isLoading && <LoadingState />}
        {error && <ErrorState message={String((error as Error).message)} />}
        {services && services.length === 0 && <EmptyState />}

        {services && services.length > 0 && (
          <div className="flex flex-col gap-2">
            {services.map((svc) => (
              <ServiceRow
                key={svc.instance_id}
                svc={svc}
                expanded={expandedId === svc.instance_id}
                onToggle={() =>
                  setExpandedId(expandedId === svc.instance_id ? null : svc.instance_id)
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ServiceRow({
  svc, expanded, onToggle,
}: {
  svc: CatalogService
  expanded: boolean
  onToggle: () => void
}) {
  const pct =
    svc.total_units > 0
      ? Math.min(100, Math.round((svc.used_units / svc.total_units) * 100))
      : 0
  const statusColor = statusColorFor(svc.status, pct)
  const statusLabel = svc.status
  const noPacks = svc.total_units === 0

  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
      }}
    >
      <button
        className="w-full flex items-center gap-3 text-left"
        onClick={onToggle}
        style={{ padding: '14px 16px' }}
      >
        <div
          className="flex items-center justify-center shrink-0"
          style={{
            width: 36, height: 36, borderRadius: 6,
            background: 'var(--accent-glow)',
            color: 'var(--accent)',
          }}
        >
          <Activity size={18} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div
              className="font-medium text-[14px] truncate"
              style={{ color: 'var(--fg)' }}
            >
              {svc.instance_name}
            </div>
            <span
              className="text-[11px] px-1.5 py-0.5 rounded"
              style={{
                background: 'var(--bg)',
                color: 'var(--muted)',
                border: '1px solid var(--border)',
              }}
            >
              {svc.category || '—'}
            </span>
            <span
              className="text-[11px] px-1.5 py-0.5 rounded"
              style={{
                background: statusColor.bg, color: statusColor.fg,
              }}
            >
              {statusLabel}
            </span>
          </div>

          <div className="flex items-center gap-3 mt-2">
            <div
              className="flex-1 h-1.5 rounded-full overflow-hidden"
              style={{ background: 'var(--border)' }}
            >
              <div
                className="h-full transition-all"
                style={{
                  width: `${pct}%`,
                  background: pct >= 95 ? 'var(--accent)'
                           : pct >= 80 ? 'var(--warn)'
                           : 'var(--accent-2)',
                }}
              />
            </div>
            <div
              className="text-[12px] tabular-nums shrink-0"
              style={{ color: 'var(--muted)' }}
            >
              {noPacks
                ? `${svc.active_grants}/${svc.total_grants} 授权，暂无资源包`
                : `${fmt(svc.used_units)} / ${fmt(svc.total_units)} ${svc.meter_dim || ''}`}
            </div>
          </div>
        </div>

        <ChevronRight
          size={16}
          className={`shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`}
          style={{ color: 'var(--muted)' }}
        />
      </button>

      {expanded && (
        <div
          style={{
            padding: '12px 16px 16px 16px',
            borderTop: '1px solid var(--border)',
            color: 'var(--muted)',
            fontSize: 13,
          }}
        >
          <div className="grid grid-cols-3 gap-4">
            <Stat label="总量" value={fmt(svc.total_units)} />
            <Stat label="已用" value={fmt(svc.used_units)} />
            <Stat label="剩余" value={fmt(svc.remaining_units)} />
          </div>
          <div className="mt-3 text-[12px]">
            Playground + 资源包管理即将上线 (Lane F)。当前视图只做只读总览。
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ color: 'var(--muted)', fontSize: 11 }}>{label}</div>
      <div style={{ color: 'var(--fg)', fontSize: 16, fontWeight: 500 }}>{value}</div>
    </div>
  )
}

function statusColorFor(status: string, pct: number) {
  if (status === 'inactive') return { bg: 'rgba(200,200,200,0.15)', fg: 'var(--muted)' }
  if (pct >= 95) return { bg: 'var(--accent-glow)', fg: 'var(--accent)' }
  if (pct >= 80) return { bg: 'rgba(250,200,70,0.15)', fg: 'var(--warn)' }
  return { bg: 'rgba(80,200,160,0.15)', fg: 'var(--accent-2)' }
}

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function LoadingState() {
  return (
    <div className="text-center py-12" style={{ color: 'var(--muted)' }}>
      加载中…
    </div>
  )
}

function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="flex items-center gap-2 p-4 rounded"
      style={{
        background: 'var(--accent-glow)',
        border: '1px solid var(--accent)',
        color: 'var(--accent)',
        fontSize: 13,
      }}
    >
      <AlertTriangle size={16} />
      {message}
    </div>
  )
}

function EmptyState() {
  return (
    <div
      className="text-center py-12"
      style={{
        background: 'var(--bg-accent)',
        border: '1px dashed var(--border)',
        borderRadius: 8,
        color: 'var(--muted)',
      }}
    >
      <div className="text-[14px]">暂无已开通服务</div>
      <div className="text-[12px] mt-1.5">
        请到 API Management 页面为此 Key 授权服务实例
      </div>
    </div>
  )
}
