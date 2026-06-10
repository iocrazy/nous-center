// 历史出图画廊(spec 2026-06-09 run-history PR-B,借鉴 Infinite-Canvas history)。
// 拉最近 task,筛出带 output_thumbnails 的出图记录 → 网格画廊;hover 卡片可「重跑」
// (复用 PR-A:按 workflow_id 找服务跳 Playground 回填 input_json)/「删除」;点图开 lightbox。
import { useMemo, useState } from 'react'
import { X, RefreshCw, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { usePanelStore } from '../../stores/panel'
import { confirmDialog } from '../../stores/confirm'
import { useImageTasks, useDeleteTask, type ExecutionTask } from '../../api/tasks'
import { useServices } from '../../api/services'

export default function HistoryOverlay() {
  const setOverlay = usePanelStore((s) => s.setOverlay)
  const navigate = useNavigate()
  const { data: tasks, isLoading } = useImageTasks()
  const { data: services } = useServices()
  const del = useDeleteTask()
  const [lightbox, setLightbox] = useState<string | null>(null)

  const items = useMemo(
    () => (tasks ?? []).filter((t) => (t.output_thumbnails?.length ?? 0) > 0),
    [tasks],
  )

  const rerun = (t: ExecutionTask) => {
    const svc = services?.find((s) => !!s.workflow_id && s.workflow_id === t.workflow_id)
    if (!svc) return
    setOverlay(null)
    navigate(`/services/${svc.id}`, { state: { rerunInputs: t.input_json } })
  }

  return (
    <div className="absolute inset-0 z-30 flex flex-col" style={{ background: 'var(--bg)', overflow: 'hidden' }}>
      {/* header */}
      <div
        className="shrink-0 flex items-center gap-3"
        style={{ padding: '12px 18px', borderBottom: '1px solid var(--border)' }}
      >
        <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>历史出图</span>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>{items.length} 张</span>
        <button
          type="button"
          onClick={() => setOverlay(null)}
          aria-label="关闭"
          style={{ marginLeft: 'auto', background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer' }}
        >
          <X size={18} />
        </button>
      </div>

      {/* grid */}
      <div style={{ flex: 1, overflow: 'auto', padding: 18 }}>
        {isLoading ? (
          <div style={{ color: 'var(--muted)', fontSize: 12, textAlign: 'center', padding: 40 }}>加载中…</div>
        ) : items.length === 0 ? (
          <div style={{ color: 'var(--muted)', fontSize: 12, textAlign: 'center', padding: 60 }}>
            还没有出图记录 — 到创作台或服务 Playground 跑一次就会出现在这里。
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: 14,
            }}
          >
            {items.map((t) => {
              const thumb = t.output_thumbnails![0]
              const canRerun = !!services?.find((s) => !!s.workflow_id && s.workflow_id === t.workflow_id)
              return (
                <Card
                  key={t.id}
                  task={t}
                  thumb={thumb}
                  canRerun={canRerun}
                  onOpen={() => setLightbox(thumb)}
                  onRerun={() => rerun(t)}
                  onDelete={async () => {
                    if (await confirmDialog({ message: '删除这条出图记录?', danger: true, confirmText: '删除' })) del.mutate(t.id)
                  }}
                />
              )
            })}
          </div>
        )}
      </div>

      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          style={{
            position: 'fixed', inset: 0, zIndex: 60, background: 'rgba(0,0,0,0.85)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40, cursor: 'zoom-out',
          }}
        >
          <img src={lightbox} alt="" style={{ maxWidth: '92vw', maxHeight: '92vh', borderRadius: 6, objectFit: 'contain' }} />
        </div>
      )}
    </div>
  )
}

function Card({
  task, thumb, canRerun, onOpen, onRerun, onDelete,
}: {
  task: ExecutionTask
  thumb: string
  canRerun: boolean
  onOpen: () => void
  onRerun: () => void
  onDelete: () => void
}) {
  const [hover, setHover] = useState(false)
  const summary = task.input_json ? JSON.stringify(task.input_json) : (task.workflow_name || '—')
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden',
        background: 'var(--card, var(--bg-accent))',
      }}
    >
      <div style={{ position: 'relative', aspectRatio: '1 / 1', background: 'var(--bg)', cursor: 'zoom-in' }} onClick={onOpen}>
        <img src={thumb} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        {(task.output_thumbnails?.length ?? 0) > 1 && (
          <span style={{
            position: 'absolute', top: 6, right: 6, fontSize: 10, padding: '1px 6px',
            borderRadius: 8, background: 'rgba(0,0,0,0.6)', color: '#fff',
          }}>
            ×{task.output_thumbnails!.length}
          </span>
        )}
        {hover && (
          <div style={{ position: 'absolute', top: 6, left: 6, display: 'flex', gap: 6 }}>
            {canRerun && (
              <IconAction title="重跑(相同参数)" onClick={(e) => { e.stopPropagation(); onRerun() }}>
                <RefreshCw size={13} />
              </IconAction>
            )}
            <IconAction title="删除" danger onClick={(e) => { e.stopPropagation(); onDelete() }}>
              <Trash2 size={13} />
            </IconAction>
          </div>
        )}
      </div>
      <div style={{ padding: '7px 9px' }}>
        <div style={{
          fontSize: 11, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }} title={summary}>
          {summary}
        </div>
        <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3 }}>
          {task.workflow_name} · {new Date(task.created_at).toLocaleString()}
        </div>
      </div>
    </div>
  )
}

function IconAction({
  children, onClick, title, danger,
}: {
  children: React.ReactNode
  onClick: (e: React.MouseEvent) => void
  title: string
  danger?: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 26, height: 26, borderRadius: 5, border: 'none', cursor: 'pointer',
        background: 'rgba(0,0,0,0.6)', color: danger ? '#f87171' : '#fff',
      }}
    >
      {children}
    </button>
  )
}
