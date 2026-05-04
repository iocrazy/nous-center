import { useMemo, useState } from 'react'
import {
  AlertCircle,
  Ban,
  CheckCircle2,
  Clock,
  Image as ImageIcon,
  Loader2,
  RotateCcw,
  Trash2,
  X,
} from 'lucide-react'
import {
  useCancelTask,
  useDeleteTask,
  useRetryTask,
  useTasks,
  type ExecutionTask,
} from '../../api/tasks'

// m15 mockup 对齐：460px 抽屉、3 tab（活跃/最近/失败）、汇总数字、
// 每条 task = spinner + 状态徽章 + 进度条 + 操作按钮组。
// 详情按钮（输出/日志）暂时 placeholder，后续接通。

type TabId = 'active' | 'recent' | 'failed'

const TAB_DEFS: { id: TabId; label: string; match: (t: ExecutionTask) => boolean }[] = [
  { id: 'active', label: '活跃', match: (t) => t.status === 'running' || t.status === 'queued' },
  { id: 'recent', label: '最近', match: (t) => t.status === 'completed' || t.status === 'cancelled' },
  { id: 'failed', label: '失败', match: (t) => t.status === 'failed' },
]

const STATUS_STYLE: Record<string, { label: string; bg: string; color: string; border?: string }> = {
  running: {
    label: '运行中',
    bg: 'rgba(59,130,246,0.15)',
    color: 'var(--info, #3b82f6)',
  },
  queued: {
    label: '排队',
    bg: 'var(--bg-accent)',
    color: 'var(--muted)',
    border: '1px solid var(--border)',
  },
  completed: {
    label: '已完成',
    bg: 'rgba(34,197,94,0.15)',
    color: 'var(--accent-2, #22c55e)',
  },
  failed: {
    label: '失败',
    bg: 'rgba(239,68,68,0.15)',
    color: 'var(--accent, #ef4444)',
  },
  cancelled: {
    label: '已取消',
    bg: 'var(--bg-accent)',
    color: 'var(--muted)',
    border: '1px solid var(--border)',
  },
}

export default function TaskPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: tasks } = useTasks()
  const [tab, setTab] = useState<TabId>('active')

  const counts = useMemo(() => {
    const out = { running: 0, queued: 0, completed: 0, failed: 0, cancelled: 0 }
    for (const t of tasks ?? []) {
      const k = t.status as keyof typeof out
      if (k in out) out[k]++
    }
    return out
  }, [tasks])

  const tabCounts: Record<TabId, number> = {
    active: counts.running + counts.queued,
    recent: counts.completed + counts.cancelled,
    failed: counts.failed,
  }

  const visibleTasks = useMemo(() => {
    const matcher = TAB_DEFS.find((t) => t.id === tab)?.match ?? (() => true)
    return (tasks ?? []).filter(matcher)
  }, [tasks, tab])

  if (!open) return null

  return (
    <>
      {/* 半透明遮罩 — m15 mockup 用 rgba(0,0,0,.4) 让背景画布失焦 */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.35)',
          zIndex: 49,
        }}
      />

      <aside
        style={{
          position: 'fixed',
          right: 0,
          top: 0,
          bottom: 0,
          width: 460,
          background: 'var(--bg-accent)',
          borderLeft: '1px solid var(--border)',
          boxShadow: '-8px 0 24px rgba(0,0,0,0.3)',
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* header */}
        <div
          style={{
            padding: '14px 18px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <h2 style={{ flex: 1, fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            任务面板
          </h2>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>
            {counts.running} 运行中 · {counts.queued} 排队 · {counts.completed} 已完成
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            style={{
              width: 24,
              height: 24,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              color: 'var(--muted)',
              borderRadius: 4,
              cursor: 'pointer',
            }}
          >
            <X size={14} />
          </button>
        </div>

        {/* tabs */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--border)',
            padding: '0 18px',
          }}
        >
          {TAB_DEFS.map((t) => {
            const active = tab === t.id
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                style={{
                  padding: '10px 14px',
                  fontSize: 12,
                  background: 'transparent',
                  border: 'none',
                  borderBottom: '2px solid',
                  borderBottomColor: active ? 'var(--accent)' : 'transparent',
                  color: active ? 'var(--text)' : 'var(--muted)',
                  cursor: 'pointer',
                }}
              >
                {t.label} {tabCounts[t.id]}
              </button>
            )
          })}
        </div>

        {/* body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 18px' }}>
          {visibleTasks.length === 0 ? (
            <Empty tab={tab} />
          ) : (
            visibleTasks.map((t) => <TaskCard key={t.id} task={t} />)
          )}
        </div>
      </aside>
    </>
  )
}

// ---------- subviews ----------

function TaskCard({ task }: { task: ExecutionTask }) {
  const cancelTask = useCancelTask()
  const retryTask = useRetryTask()
  const deleteTask = useDeleteTask()

  const cfg = STATUS_STYLE[task.status] ?? STATUS_STYLE.queued
  const isRunning = task.status === 'running'
  const isFailed = task.status === 'failed'
  const isCancelled = task.status === 'cancelled'
  const canCancel = task.status === 'queued' || task.status === 'running'
  const canRetry = isFailed || isCancelled
  const progress = task.nodes_total > 0 ? (task.nodes_done / task.nodes_total) * 100 : 0

  return (
    <div
      style={{
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '12px 14px',
        marginBottom: 10,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <StatusIcon status={task.status} />
        {task.task_type === 'image' && (
          <ImageIcon
            size={12}
            aria-label="图像任务"
            style={{ color: 'var(--info, #3b82f6)', flexShrink: 0 }}
          />
        )}
        <div
          style={{
            flex: 1,
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--text)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {task.workflow_name || '未命名任务'}
        </div>
        <span
          style={{
            fontSize: 11,
            padding: '2px 7px',
            borderRadius: 3,
            background: cfg.bg,
            color: cfg.color,
            border: cfg.border,
          }}
        >
          {cfg.label}
        </span>
      </div>

      <div
        style={{
          display: 'flex',
          gap: 12,
          fontSize: 11,
          color: 'var(--muted)',
          marginTop: 4,
          flexWrap: 'wrap',
        }}
      >
        <span style={{ fontFamily: 'var(--mono, monospace)' }}>#{shortId(task.id)}</span>
        <span>started {formatTime(task.created_at)}</span>
        {task.nodes_total > 0 && (
          <span>
            步骤 {task.nodes_done} / {task.nodes_total}
          </span>
        )}
        {task.image_width && task.image_height && (
          <span>
            {task.image_width}×{task.image_height}
          </span>
        )}
        {task.duration_ms != null && task.duration_ms > 0 && (
          <span>耗时 {formatDuration(task.duration_ms)}</span>
        )}
      </div>

      {(isRunning || (task.nodes_total > 0 && !isCancelled)) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
          <div
            style={{
              flex: 1,
              height: 4,
              background: 'var(--bg-accent)',
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: '100%',
                background: isFailed ? 'var(--accent)' : 'var(--accent-2, #22c55e)',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
          <span
            style={{
              fontSize: 11,
              color: 'var(--muted)',
              minWidth: 36,
              textAlign: 'right',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {Math.round(progress)}%
          </span>
        </div>
      )}

      {task.error && (
        <div
          style={{
            marginTop: 8,
            padding: '6px 10px',
            borderRadius: 4,
            background: 'rgba(239,68,68,0.08)',
            color: 'var(--accent, #ef4444)',
            fontSize: 11,
            wordBreak: 'break-all',
            maxHeight: 80,
            overflowY: 'auto',
            border: '1px solid rgba(239,68,68,0.25)',
          }}
        >
          {task.error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
        {canCancel && (
          <ActionBtn
            onClick={() => cancelTask.mutate(task.id)}
            disabled={cancelTask.isPending}
            icon={<Ban size={11} />}
            label="取消"
            danger
          />
        )}
        {canRetry && (
          <ActionBtn
            onClick={() => retryTask.mutate(task.id)}
            disabled={retryTask.isPending}
            icon={<RotateCcw size={11} />}
            label="重试"
          />
        )}
        <ActionBtn
          onClick={() => deleteTask.mutate(task.id)}
          disabled={deleteTask.isPending}
          icon={<Trash2 size={11} />}
          label="删除"
        />
      </div>
    </div>
  )
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'running') {
    return (
      <Loader2
        size={14}
        style={{
          color: 'var(--info, #3b82f6)',
          animation: 'spin 1.5s linear infinite',
          flexShrink: 0,
        }}
      />
    )
  }
  if (status === 'queued') {
    return <Clock size={14} style={{ color: 'var(--muted)', flexShrink: 0 }} />
  }
  if (status === 'completed') {
    return (
      <CheckCircle2 size={14} style={{ color: 'var(--accent-2, #22c55e)', flexShrink: 0 }} />
    )
  }
  if (status === 'failed') {
    return <AlertCircle size={14} style={{ color: 'var(--accent, #ef4444)', flexShrink: 0 }} />
  }
  return <Clock size={14} style={{ color: 'var(--muted)', flexShrink: 0 }} />
}

function ActionBtn({
  onClick,
  disabled,
  icon,
  label,
  danger,
}: {
  onClick: () => void
  disabled?: boolean
  icon: React.ReactNode
  label: string
  danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      disabled={disabled}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '4px 10px',
        fontSize: 11,
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: 'transparent',
        color: danger ? 'var(--accent, #ef4444)' : 'var(--text)',
        cursor: disabled ? 'wait' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {icon}
      {label}
    </button>
  )
}

function Empty({ tab }: { tab: TabId }) {
  const map: Record<TabId, string> = {
    active: '当前没有运行中或排队的任务。',
    recent: '没有最近完成的任务。',
    failed: '没有失败任务（很好）。',
  }
  return (
    <div
      style={{
        padding: 32,
        textAlign: 'center',
        fontSize: 12,
        color: 'var(--muted)',
      }}
    >
      {map[tab]}
    </div>
  )
}

// ---------- helpers ----------

function shortId(id: string): string {
  if (id.length <= 12) return id
  return id.slice(0, 8)
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('zh-CN', { hour12: false })
  } catch {
    return iso
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m${Math.round(s - m * 60)}s`
}
