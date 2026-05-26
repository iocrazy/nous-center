/**
 * TaskMenuButton — Topbar 全局任务入口(对齐 ComfyUI 截图 1:1)。
 *
 * 渐进披露(用户截图明确指出 popover 和 drawer 不该重复):
 *   - chip:`● N 个活动任务`(蓝点突出)/ `任务`(idle)。
 *   - **紧凑 popover**:只显**当前任务** + live preview + **双进度条**(全部:N% / 节点:M%)
 *     + 行内中止;**不**列已完成(那是 drawer 的事)。
 *   - 「查看全部 →」打开 TaskPanel(完整管理,见 PR-5)。
 *
 * 双进度条对齐 ComfyUI「全部:59% / 自定义采样器(高级):12%」。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronRight, Zap, XCircle } from 'lucide-react'
import { useCancelTask, useTasks, type ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

const RUNNING = new Set<ExecutionTask['status']>(['queued', 'running'])

export default function TaskMenuButton() {
  const { data: tasks } = useTasks()
  const toggleTaskPanel = useExecutionStore((s) => s.toggleTaskPanel)
  const wfProgress = useExecutionStore((s) => s.progress)
  const nodeProgress = useExecutionStore((s) => s.currentNodeProgress)
  const nodeStep = useExecutionStore((s) => s.currentNodeStep)
  const nodeType = useExecutionStore((s) => s.currentNodeType)
  const previewUrl = useExecutionStore((s) => s.latestPreviewUrl)
  const cancel = useCancelTask()
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  const running = useMemo(() => (tasks ?? []).filter((t) => RUNNING.has(t.status)), [tasks])
  const current = running[0]
  const isBusy = running.length > 0

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
          background: open ? 'var(--bg-hover)' : isBusy ? 'var(--accent-subtle)' : 'transparent',
          border: '1px solid ' + (isBusy ? 'var(--info)' : 'var(--border)'),
          borderRadius: 6,
          color: isBusy ? 'var(--info)' : 'var(--text)',
          cursor: 'pointer',
          transition: 'background 0.15s, border-color 0.15s',
        }}
      >
        {isBusy ? (
          <>
            <span
              style={{
                width: 7, height: 7, borderRadius: '50%',
                background: 'var(--info)', boxShadow: '0 0 6px var(--info)',
                flexShrink: 0,
              }}
            />
            <span>{running.length} 个活动任务</span>
          </>
        ) : (
          <>
            <span
              style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'var(--muted)', flexShrink: 0,
              }}
            />
            <span>任务</span>
          </>
        )}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="任务概览"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            right: 0,
            width: 320,
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            boxShadow: 'var(--shadow-lg)',
            zIndex: 100,
            overflow: 'hidden',
          }}
        >
          {current ? (
            <ActiveTaskPanel
              task={current}
              wfProgress={wfProgress}
              nodeProgress={nodeProgress}
              nodeStep={nodeStep}
              nodeType={nodeType}
              previewUrl={previewUrl}
              onCancel={() => cancel.mutate(current.id)}
              canceling={cancel.isPending}
            />
          ) : (
            <div style={{ padding: 24, textAlign: 'center', fontSize: 11, color: 'var(--muted)' }}>
              暂无运行中任务
            </div>
          )}
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
              width: '100%',
              padding: '8px 10px',
              borderTop: '1px solid var(--border)',
              background: 'transparent',
              border: 'none',
              fontSize: 11,
              color: 'var(--accent)',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            查看全部 <ChevronRight size={11} />
          </button>
        </div>
      )}
    </div>
  )
}

function ActiveTaskPanel({
  task,
  wfProgress,
  nodeProgress,
  nodeStep,
  nodeType,
  previewUrl,
  onCancel,
  canceling,
}: {
  task: ExecutionTask
  wfProgress: number
  nodeProgress: number | null
  nodeStep: { done: number; total: number } | null
  nodeType: string | null
  previewUrl: string | null
  onCancel: () => void
  canceling: boolean
}) {
  // 工作流总 % 来自 backend nodes_done/total(节点级);若 backend 没发就用 execution store 的 wfProgress。
  const wfPct = task.nodes_total > 0
    ? Math.round((task.nodes_done / task.nodes_total) * 100)
    : Math.round(wfProgress)
  // 节点内 step 进度(PR-3 / PR-F),如 KSampler 第 12/25 步 → 48%。
  const nodePct = nodeProgress ?? 0
  return (
    <div style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* 标题行:闪电图标 + 工作流名 + 中止 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Zap size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span
          style={{
            flex: 1,
            fontSize: 12,
            fontWeight: 500,
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
          onClick={onCancel}
          disabled={canceling}
          aria-label="中止任务"
          title="中止任务"
          style={{
            width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'var(--err, #ef4444)', borderRadius: 3,
          }}
        >
          <XCircle size={13} />
        </button>
      </div>
      {/* live preview(latent → RGB JPEG;PR-F)。出图过程图慢慢长出来。 */}
      {previewUrl && (
        <div
          style={{
            width: '100%',
            display: 'flex',
            justifyContent: 'center',
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: 4,
          }}
        >
          <img
            src={previewUrl}
            alt="latent preview"
            style={{
              maxHeight: 120,
              maxWidth: '100%',
              borderRadius: 3,
              imageRendering: 'pixelated',
            }}
          />
        </div>
      )}
      {/* 双进度条(对齐 ComfyUI):全部 + 当前节点 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <ProgressBar label={`全部:${wfPct}%`} pct={wfPct} thick />
        {nodeProgress !== null && (
          <ProgressBar
            label={`${nodeType || '当前节点'}${nodeStep ? ` ${nodeStep.done}/${nodeStep.total}` : ''} · ${nodePct}%`}
            pct={nodePct}
            thick={false}
          />
        )}
      </div>
    </div>
  )
}

function ProgressBar({ label, pct, thick }: { label: string; pct: number; thick: boolean }) {
  return (
    <div>
      <div
        style={{
          position: 'relative',
          height: thick ? 18 : 10,
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: thick ? 4 : 3,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: thick ? 'var(--info)' : 'var(--accent-2, #14b8a6)',
            transition: 'width 0.3s ease',
          }}
        />
        <span
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            paddingLeft: 8,
            fontSize: thick ? 10 : 9,
            fontWeight: thick ? 600 : 500,
            color: 'var(--text-strong, white)',
            mixBlendMode: 'difference',
          }}
        >
          {label}
        </span>
      </div>
    </div>
  )
}
