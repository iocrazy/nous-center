import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  Ban,
  CheckCircle2,
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
import { useRunners, type RunnerInfo } from '../../api/runners'

// Lane I（spec §6）：TaskPanel 从「3-tab 扁平列表」重构为 Buildkite 风：
//   - 顶部 per-runner 泳道区（视觉 hero）—— 每条泳道 = 一个 GPU runner 的
//     当前任务 + 进度条 + 排队数（可展开）+ 异常态内联。
//   - 下方「最近完成」列表 —— image 任务带输出缩略图。
// 响应式：<768px 抽屉变全屏、泳道堆叠。a11y：折叠 toggle 用真 button +
// aria-expanded、状态文字始终伴随色点、动作键盘可达。

const TERMINAL: ReadonlySet<ExecutionTask['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
])

/** <768px 判定 —— 抽屉全屏 + 泳道堆叠（DD7）。 */
function useIsNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    typeof window !== 'undefined' ? window.innerWidth < 768 : false,
  )
  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth < 768)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return narrow
}

export default function TaskPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: tasks } = useTasks()
  const { data: runners } = useRunners()
  const isNarrow = useIsNarrow()

  const recent = useMemo(
    () => (tasks ?? []).filter((t) => TERMINAL.has(t.status)),
    [tasks],
  )

  if (!open) return null

  return (
    <>
      {/* 半透明遮罩 —— 点击关闭。<768px 全屏抽屉时遮罩仍铺满（点不到也无妨）。 */}
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 49 }}
      />

      <aside
        style={{
          position: 'fixed',
          right: 0,
          top: 0,
          bottom: 0,
          // DD7 响应式：<768px 全屏，否则 460px 抽屉。
          width: isNarrow ? '100vw' : 460,
          background: 'var(--bg-accent)',
          borderLeft: isNarrow ? 'none' : '1px solid var(--border)',
          boxShadow: isNarrow ? 'none' : '-8px 0 24px rgba(0,0,0,0.3)',
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

        {/* body —— 泳道区（hero）+ 最近完成 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 18px' }}>
          {/* per-runner 泳道区 */}
          <section aria-label="GPU runner 泳道" style={{ marginBottom: 18 }}>
            {(runners ?? []).length === 0 ? (
              <div
                style={{
                  padding: 24,
                  textAlign: 'center',
                  fontSize: 12,
                  color: 'var(--muted)',
                  border: '1px dashed var(--border)',
                  borderRadius: 6,
                }}
              >
                暂无 runner 数据（调度器未就绪或端点未上线）
              </div>
            ) : (
              <div
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  overflow: 'hidden',
                }}
              >
                {(runners ?? []).map((r) => (
                  <RunnerLane key={r.id} runner={r} />
                ))}
              </div>
            )}
          </section>

          {/* 最近完成 */}
          <section aria-label="最近完成">
            <div
              style={{
                fontSize: 11,
                color: 'var(--muted)',
                textTransform: 'uppercase',
                letterSpacing: 0.5,
                marginBottom: 8,
              }}
            >
              最近完成
            </div>
            {recent.length === 0 ? (
              <div
                style={{
                  padding: 24,
                  textAlign: 'center',
                  fontSize: 12,
                  color: 'var(--muted)',
                }}
              >
                还没有完成的任务。
              </div>
            ) : (
              recent.map((t) => <RecentRow key={t.id} task={t} />)
            )}
          </section>
        </div>
      </aside>
    </>
  )
}

// ---------- runner 泳道（Task 9 加异常态 + Task 10 加排队展开）----------

function RunnerLane({ runner }: { runner: RunnerInfo }) {
  return (
    <div
      style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <RunnerStateDot state={runner.state} />
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
          {runner.label}
        </span>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>({runner.role})</span>
        <span style={{ flex: 1 }} />
        {/* 状态文字始终伴随色点（a11y：色盲不能只靠颜色）。 */}
        <RunnerStateText runner={runner} />
      </div>

      {runner.state === 'busy' && runner.current_task && (
        <div style={{ marginTop: 8 }}>
          <div
            style={{
              fontSize: 12,
              color: 'var(--text)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {runner.current_task.workflow_name}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
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
                  width: `${Math.round((runner.current_task.progress ?? 0) * 100)}%`,
                  height: '100%',
                  background: 'var(--accent-2, #22c55e)',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
            {runner.current_task.detail && (
              <span
                style={{
                  fontSize: 11,
                  color: 'var(--muted)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {runner.current_task.detail}
              </span>
            )}
          </div>
        </div>
      )}

      {/* 排队数 —— Task 10 替换为可展开的 QueueExpand。 */}
      {runner.queue.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
          排队 {runner.queue.length}
        </div>
      )}
    </div>
  )
}

function RunnerStateDot({ state }: { state: RunnerInfo['state'] }) {
  const color =
    state === 'busy'
      ? 'var(--accent-2, #22c55e)'
      : state === 'restarting'
        ? 'var(--warn, #f59e0b)'
        : state === 'load_failed'
          ? 'var(--accent, #ef4444)'
          : 'var(--muted)'
  return (
    <span
      aria-hidden="true"
      style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
        flexShrink: 0,
        // restarting 态脉冲（Task 9 接入 keyframes）。
        animation: state === 'restarting' ? 'pulse 1.2s ease-in-out infinite' : undefined,
      }}
    />
  )
}

function RunnerStateText({ runner }: { runner: RunnerInfo }) {
  // Task 9 把 restarting / load_failed 的文案 + Retry 按钮做全；
  // 本 Task 先给 idle / busy 的文字。
  if (runner.state === 'idle') {
    return <span style={{ fontSize: 11, color: 'var(--muted)' }}>idle</span>
  }
  if (runner.state === 'busy') {
    return <span style={{ fontSize: 11, color: 'var(--accent-2, #22c55e)' }}>busy</span>
  }
  if (runner.state === 'restarting') {
    return <span style={{ fontSize: 11, color: 'var(--warn, #f59e0b)' }}>重启中</span>
  }
  return <span style={{ fontSize: 11, color: 'var(--accent, #ef4444)' }}>加载失败</span>
}

// ---------- 最近完成列表行（Task 11 加缩略图）----------

function RecentRow({ task }: { task: ExecutionTask }) {
  const cancelTask = useCancelTask()
  const retryTask = useRetryTask()
  const deleteTask = useDeleteTask()

  const isFailed = task.status === 'failed'
  const isCancelled = task.status === 'cancelled'
  const canRetry = isFailed || isCancelled

  return (
    <div
      style={{
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '10px 12px',
        marginBottom: 8,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}
    >
      {/* Task 11：image 任务这里放缩略图，否则状态图标。 */}
      <RecentLeading task={task} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
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
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
          {statusLabel(task.status)}
          {task.duration_ms != null && task.duration_ms > 0
            ? ` · ${formatDuration(task.duration_ms)}`
            : ''}
        </div>
        {task.error && (
          <div
            style={{
              marginTop: 6,
              fontSize: 11,
              color: 'var(--accent, #ef4444)',
              wordBreak: 'break-all',
            }}
          >
            {task.error}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
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
        {task.status === 'queued' || task.status === 'running' ? (
          <ActionBtn
            onClick={() => cancelTask.mutate(task.id)}
            disabled={cancelTask.isPending}
            icon={<Ban size={11} />}
            label="取消"
            danger
          />
        ) : null}
      </div>
    </div>
  )
}

function RecentLeading({ task }: { task: ExecutionTask }) {
  // Task 11 用 ImageThumb 替换；本 Task 先放状态图标。
  if (task.status === 'completed') {
    return <CheckCircle2 size={16} style={{ color: 'var(--accent-2, #22c55e)', flexShrink: 0 }} />
  }
  if (task.status === 'failed') {
    return <AlertCircle size={16} style={{ color: 'var(--accent, #ef4444)', flexShrink: 0 }} />
  }
  return <Loader2 size={16} style={{ color: 'var(--muted)', flexShrink: 0 }} />
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
      aria-label={label}
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

// ---------- helpers ----------

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    running: '运行中',
    queued: '排队',
  }
  return map[status] ?? status
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m${Math.round(s - m * 60)}s`
}

// 占位：ImageIcon 在 Task 11 被 ImageThumb 用到，先 re-export 防 unused-import。
export const _ImageIcon = ImageIcon
