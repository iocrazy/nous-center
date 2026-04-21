import { AppWindow, AlertTriangle } from 'lucide-react'
import { useApps, type WorkflowApp } from '../../api/apps'

export default function AppsOverlay() {
  const { data: apps, isLoading, error } = useApps()

  return (
    <div
      className="absolute inset-0 overflow-y-auto z-[16]"
      style={{ background: 'var(--bg)' }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
        <div className="mb-5 flex items-center justify-between">
          <div>
            <h1 className="text-[20px] font-semibold" style={{ color: 'var(--fg)' }}>
              应用
            </h1>
            <p className="text-[13px] mt-1" style={{ color: 'var(--muted)' }}>
              已发布的 Workflow Apps — 每个 App 都有独立的外部调用 URL
            </p>
          </div>
        </div>

        {isLoading && (
          <div className="text-center py-12" style={{ color: 'var(--muted)' }}>
            加载中…
          </div>
        )}
        {error && (
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
            {(error as Error).message}
          </div>
        )}

        {apps && apps.length === 0 && (
          <div
            className="text-center py-12"
            style={{
              background: 'var(--bg-accent)',
              border: '1px dashed var(--border)',
              borderRadius: 8,
              color: 'var(--muted)',
            }}
          >
            <div className="text-[14px]">还没有发布的 App</div>
            <div className="text-[12px] mt-1.5">
              在 Workflow 编辑器里点"发布为 App"开始
            </div>
          </div>
        )}

        {apps && apps.length > 0 && (
          <div className="grid grid-cols-3 gap-3">
            {apps.map((app) => (
              <AppCard key={app.id} app={app} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function AppCard({ app }: { app: WorkflowApp }) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: 16,
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <div
          className="flex items-center justify-center shrink-0"
          style={{
            width: 32, height: 32, borderRadius: 6,
            background: 'var(--accent-glow)',
            color: 'var(--accent)',
          }}
        >
          <AppWindow size={16} />
        </div>
        <div className="flex-1 min-w-0">
          <div
            className="font-medium text-[14px] truncate"
            style={{ color: 'var(--fg)' }}
          >
            {app.display_name || app.name}
          </div>
          <div
            className="text-[11px] font-mono truncate"
            style={{ color: 'var(--muted)' }}
          >
            {app.name}
          </div>
        </div>
        <span
          className="text-[10px] px-1.5 py-0.5 rounded"
          style={{
            background: app.active ? 'rgba(80,200,160,0.15)' : 'rgba(120,120,140,0.15)',
            color: app.active ? 'var(--accent-2)' : 'var(--muted)',
          }}
        >
          {app.active ? 'active' : 'inactive'}
        </span>
      </div>

      {app.description && (
        <p
          className="text-[12px] mt-2 line-clamp-2"
          style={{ color: 'var(--muted)' }}
        >
          {app.description}
        </p>
      )}

      <div
        className="flex items-center gap-3 mt-3 pt-2 text-[11px]"
        style={{ borderTop: '1px solid var(--border)', color: 'var(--muted)' }}
      >
        <span>{app.call_count || 0} 次调用</span>
        <span>•</span>
        <span>{app.exposed_inputs.length} 入参</span>
        <span>•</span>
        <span>{app.exposed_outputs.length} 出参</span>
      </div>
    </div>
  )
}
