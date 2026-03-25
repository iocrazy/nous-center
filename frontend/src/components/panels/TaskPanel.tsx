import { useState } from 'react'
import { X, RotateCcw, Trash2, Ban, ChevronDown, ChevronRight } from 'lucide-react'
import { useTasks, useCancelTask, useRetryTask, useDeleteTask, type ExecutionTask } from '../../api/tasks'

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

const STATUS_CONFIG: Record<string, { color: string; label: string }> = {
  queued: { color: 'var(--muted)', label: '排队' },
  running: { color: 'var(--accent)', label: '运行中' },
  completed: { color: 'var(--ok)', label: '完成' },
  failed: { color: 'var(--warn)', label: '失败' },
  cancelled: { color: 'var(--muted)', label: '取消' },
}

function StatusDot({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.queued
  return (
    <span
      style={{
        display: 'inline-block',
        width: 7,
        height: 7,
        borderRadius: '50%',
        background: cfg.color,
        flexShrink: 0,
      }}
    />
  )
}

function TaskItem({ task }: { task: ExecutionTask }) {
  const [expanded, setExpanded] = useState(false)
  const cancelTask = useCancelTask()
  const retryTask = useRetryTask()
  const deleteTask = useDeleteTask()

  const cfg = STATUS_CONFIG[task.status] ?? STATUS_CONFIG.queued
  const isRunning = task.status === 'running'
  const isFailed = task.status === 'failed'
  const isCancelled = task.status === 'cancelled'
  const canCancel = task.status === 'queued' || task.status === 'running'
  const canRetry = isFailed || isCancelled
  const progress = task.nodes_total > 0 ? (task.nodes_done / task.nodes_total) * 100 : 0

  return (
    <div
      style={{
        padding: '8px 12px',
        borderBottom: '1px solid var(--border)',
        fontSize: 11,
      }}
    >
      {/* Header row */}
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown size={10} color="var(--muted)" /> : <ChevronRight size={10} color="var(--muted)" />}
        <StatusDot status={task.status} />
        <span
          style={{
            flex: 1,
            fontWeight: 500,
            color: 'var(--text-strong)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {task.workflow_name || '未命名'}
        </span>
        <span style={{ color: 'var(--muted)', fontSize: 10, flexShrink: 0 }}>
          {formatDuration(task.duration_ms)}
        </span>
        <span style={{ color: 'var(--muted)', fontSize: 10, flexShrink: 0 }}>
          {formatRelativeTime(task.created_at)}
        </span>
      </div>

      {/* Progress bar for running tasks */}
      {isRunning && (
        <div style={{ marginTop: 6 }}>
          <div
            style={{
              height: 3,
              borderRadius: 2,
              background: 'var(--bg-hover)',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${progress}%`,
                background: 'var(--accent)',
                borderRadius: 2,
                transition: 'width 0.3s ease',
              }}
            />
          </div>
          <div style={{ marginTop: 3, color: 'var(--muted)', fontSize: 10 }}>
            {task.current_node && <span>{task.current_node} </span>}
            <span>{task.nodes_done}/{task.nodes_total}</span>
          </div>
        </div>
      )}

      {/* Expanded details */}
      {expanded && (
        <div style={{ marginTop: 6, paddingLeft: 16 }}>
          <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>
            Status: <span style={{ color: cfg.color }}>{cfg.label}</span>
            {task.nodes_total > 0 && (
              <span> | Nodes: {task.nodes_done}/{task.nodes_total}</span>
            )}
          </div>

          {task.error && (
            <div
              style={{
                marginTop: 4,
                padding: '4px 8px',
                borderRadius: 4,
                background: 'rgba(255,59,48,0.08)',
                color: 'var(--warn)',
                fontSize: 10,
                wordBreak: 'break-all',
                maxHeight: 80,
                overflowY: 'auto',
              }}
            >
              {task.error}
            </div>
          )}

          {task.result && (
            <div
              style={{
                marginTop: 4,
                padding: '4px 8px',
                borderRadius: 4,
                background: 'var(--bg-hover)',
                color: 'var(--muted)',
                fontSize: 10,
                maxHeight: 80,
                overflowY: 'auto',
                wordBreak: 'break-all',
              }}
            >
              {typeof task.result === 'string' ? task.result : JSON.stringify(task.result).slice(0, 200)}
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
            {canCancel && (
              <ActionButton
                onClick={() => cancelTask.mutate(task.id)}
                disabled={cancelTask.isPending}
                icon={<Ban size={10} />}
                label="取消"
              />
            )}
            {canRetry && (
              <ActionButton
                onClick={() => retryTask.mutate(task.id)}
                disabled={retryTask.isPending}
                icon={<RotateCcw size={10} />}
                label="重试"
              />
            )}
            <ActionButton
              onClick={() => deleteTask.mutate(task.id)}
              disabled={deleteTask.isPending}
              icon={<Trash2 size={10} />}
              label="删除"
              danger
            />
          </div>
        </div>
      )}
    </div>
  )
}

function ActionButton({
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
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      disabled={disabled}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 3,
        padding: '2px 6px',
        fontSize: 10,
        borderRadius: 3,
        border: '1px solid var(--border)',
        background: 'none',
        color: danger ? 'var(--warn)' : 'var(--muted)',
        cursor: disabled ? 'wait' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {icon}
      {label}
    </button>
  )
}

export default function TaskPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: tasks } = useTasks()

  if (!open) return null

  const runningCount = tasks?.filter((t) => t.status === 'running').length ?? 0

  return (
    <div
      style={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        width: 320,
        background: 'var(--bg)',
        borderLeft: '1px solid var(--border)',
        zIndex: 20,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '10px 12px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontWeight: 600, fontSize: 12 }}>任务列表</span>
          {tasks && tasks.length > 0 && (
            <span
              style={{
                fontSize: 10,
                padding: '1px 5px',
                borderRadius: 8,
                background: 'var(--bg-hover)',
                color: 'var(--muted)',
              }}
            >
              {tasks.length}
            </span>
          )}
          {runningCount > 0 && (
            <span
              style={{
                fontSize: 10,
                padding: '1px 5px',
                borderRadius: 8,
                background: 'rgba(0,122,255,0.15)',
                color: 'var(--accent)',
              }}
            >
              {runningCount} running
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--muted)',
            cursor: 'pointer',
            padding: 2,
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <X size={14} />
        </button>
      </div>

      {/* Task list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {tasks?.map((task) => <TaskItem key={task.id} task={task} />)}
        {(!tasks || tasks.length === 0) && (
          <div style={{ padding: 16, color: 'var(--muted)', fontSize: 11, textAlign: 'center' }}>
            暂无任务
          </div>
        )}
      </div>
    </div>
  )
}
