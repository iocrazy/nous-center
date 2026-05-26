/**
 * TaskMenuButton — Topbar 全局任务入口(Vercel deployments / Linear inbox 风渐进披露)。
 *
 * 三层披露:
 *   1. 顶栏 chip:`○ idle` / `● N 运行中`(常驻,不打断流)。
 *   2. 点开 popover:运行中 + 最近完成的紧凑列表(行内 cancel,缩略图)。
 *   3. 「查看全部 →」打开完整 TaskPanel(详情/历史管理,PR-5 抽屉)。
 *
 * Click-outside / ESC 关 popover。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCircle2,
  ChevronRight,
  Image as ImageIcon,
  Loader2,
  XCircle,
} from 'lucide-react'
import { useCancelTask, useTasks, type ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

const TERMINAL = new Set<ExecutionTask['status']>(['completed', 'failed', 'cancelled'])
const RUNNING = new Set<ExecutionTask['status']>(['queued', 'running'])
const MAX_ROWS = 6

function fmtDur(ms: number | null | undefined): string {
  if (!ms) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function statusColor(s: ExecutionTask['status']): string {
  if (s === 'completed') return 'var(--ok)'
  if (s === 'failed') return 'var(--err, #ef4444)'
  if (s === 'cancelled') return 'var(--muted)'
  if (s === 'running') return 'var(--info)'
  return 'var(--warn)'
}

export default function TaskMenuButton() {
  const { data: tasks } = useTasks()
  const toggleTaskPanel = useExecutionStore((s) => s.toggleTaskPanel)
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  const running = useMemo(() => (tasks ?? []).filter((t) => RUNNING.has(t.status)), [tasks])
  const recent = useMemo(
    () => (tasks ?? []).filter((t) => TERMINAL.has(t.status)).slice(0, MAX_ROWS),
    [tasks],
  )

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const isBusy = running.length > 0

  return (
    <div ref={wrapRef} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label="任务"
        title="任务"
        aria-haspopup="dialog"
        aria-expanded={open}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          padding: '4px 10px',
          fontSize: 11,
          fontWeight: 500,
          background: open ? 'var(--bg-hover)' : 'transparent',
          border: '1px solid ' + (isBusy ? 'var(--info)' : 'var(--border)'),
          borderRadius: 6,
          color: isBusy ? 'var(--info)' : 'var(--text)',
          cursor: 'pointer',
          transition: 'background 0.15s, border-color 0.15s',
        }}
        onMouseEnter={(e) => {
          if (!open) e.currentTarget.style.background = 'var(--bg-hover)'
        }}
        onMouseLeave={(e) => {
          if (!open) e.currentTarget.style.background = 'transparent'
        }}
      >
        {isBusy ? (
          <Loader2 size={11} className="animate-spin" />
        ) : (
          <span
            style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--muted)',
            }}
          />
        )}
        <span>{isBusy ? `${running.length} 运行中` : '任务'}</span>
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="任务概览"
          className="nowheel"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            right: 0,
            width: 320,
            maxHeight: 480,
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            boxShadow: 'var(--shadow-lg)',
            zIndex: 100,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: '8px 10px',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>
              任务概览
            </span>
            {isBusy && (
              <span
                style={{
                  padding: '1px 6px',
                  borderRadius: 10,
                  background: 'var(--accent-subtle)',
                  color: 'var(--accent)',
                  fontSize: 10,
                  fontWeight: 600,
                }}
              >
                {running.length} 运行中
              </span>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: 6 }}>
            {running.length === 0 && recent.length === 0 && (
              <div
                style={{
                  padding: 24,
                  textAlign: 'center',
                  fontSize: 11,
                  color: 'var(--muted)',
                }}
              >
                暂无任务
              </div>
            )}
            {running.map((t) => (
              <RunningRow key={t.id} task={t} />
            ))}
            {running.length > 0 && recent.length > 0 && (
              <div
                style={{
                  margin: '6px 4px',
                  padding: '0 6px',
                  fontSize: 9,
                  color: 'var(--muted)',
                  textTransform: 'uppercase',
                  letterSpacing: 0.4,
                }}
              >
                最近完成
              </div>
            )}
            {recent.map((t) => (
              <CompletedRow key={t.id} task={t} />
            ))}
          </div>

          <button
            type="button"
            onClick={() => {
              setOpen(false)
              toggleTaskPanel()
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 4,
              padding: '8px 10px',
              borderTop: '1px solid var(--border)',
              background: 'transparent',
              border: 'none',
              fontSize: 11,
              color: 'var(--accent)',
              cursor: 'pointer',
              fontWeight: 500,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            查看全部 <ChevronRight size={11} />
          </button>
        </div>
      )}
    </div>
  )
}

function RunningRow({ task }: { task: ExecutionTask }) {
  const cancel = useCancelTask()
  const pct =
    task.nodes_total > 0 ? Math.round((task.nodes_done / task.nodes_total) * 100) : 0
  return (
    <div
      style={{
        padding: '6px 8px',
        marginBottom: 2,
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 4,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Loader2 size={10} className="animate-spin" style={{ color: 'var(--info)' }} />
        <span
          style={{
            flex: 1,
            fontSize: 11,
            color: 'var(--text)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {task.workflow_name || `任务 ${String(task.id).slice(0, 8)}`}
        </span>
        <button
          type="button"
          onClick={() => cancel.mutate(task.id)}
          disabled={cancel.isPending}
          aria-label="中止"
          title="中止"
          style={{
            width: 18,
            height: 18,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'transparent',
            border: 'none',
            color: 'var(--err, #ef4444)',
            cursor: 'pointer',
            borderRadius: 3,
          }}
        >
          <XCircle size={12} />
        </button>
      </div>
      <div
        style={{
          height: 2,
          background: 'var(--border)',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: 'var(--info)',
            transition: 'width 0.3s linear',
          }}
        />
      </div>
    </div>
  )
}

function CompletedRow({ task }: { task: ExecutionTask }) {
  const thumb = task.output_thumbnails?.[0]
  return (
    <div
      style={{
        padding: 5,
        marginBottom: 2,
        background: 'transparent',
        border: '1px solid transparent',
        borderRadius: 4,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--bg-hover)'
        e.currentTarget.style.borderColor = 'var(--border)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.borderColor = 'transparent'
      }}
    >
      <div
        style={{
          width: 24,
          height: 24,
          flexShrink: 0,
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        {thumb ? (
          <img
            src={thumb}
            alt=""
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : task.task_type === 'image' ? (
          <ImageIcon size={11} style={{ color: 'var(--muted)' }} />
        ) : task.status === 'failed' ? (
          <XCircle size={11} style={{ color: 'var(--err, #ef4444)' }} />
        ) : (
          <CheckCircle2 size={11} style={{ color: 'var(--ok)' }} />
        )}
      </div>
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <span
          style={{
            fontSize: 11,
            color: 'var(--text)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {task.workflow_name || `任务 ${String(task.id).slice(0, 8)}`}
        </span>
        <span style={{ fontSize: 9, color: 'var(--muted)' }}>
          <span style={{ color: statusColor(task.status) }}>●</span> {fmtDur(task.duration_ms)}
        </span>
      </div>
    </div>
  )
}
