import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Ban,
  CheckCircle2,
  ExternalLink,
  Image as ImageIcon,
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
  X,
  XCircle,
} from 'lucide-react'
import {
  useCancelTask,
  useDeleteTask,
  useRetryTask,
  useTasks,
  type ExecutionTask,
} from '../../api/tasks'
import { usePanelStore } from '../../stores/panel'
import ContextMenu, { type MenuItem } from '../ui/ContextMenu'

/**
 * TaskPanel — 对齐 ComfyUI 「任务历史」面板风格(读用户截图复刻):
 *
 * 两种模式(panel store taskPanelMode 持久):
 *  - dock(默认):右侧 460px 抽屉,带半透明遮罩,modal-ish。
 *  - float:右下角 360x480 浮窗,无遮罩,不阻塞 canvas 操作(可同时点节点/Run)。
 *
 * Tabs:全部 / 已完成。
 * 运行中任务 = 工作流名 + 耗时 + 进度条 + cancel 按钮。
 * 已完成 = 缩略图(output_thumbnails[0])+ 标识 + 耗时 + 右键菜单(查看图片/复制ID/重试/删除)。
 */

const TERMINAL: ReadonlySet<ExecutionTask['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
])
const RUNNING: ReadonlySet<ExecutionTask['status']> = new Set(['queued', 'running'])

function fmtMs(ms: number | null | undefined): string {
  if (!ms) return '—'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function tickElapsed(createdAt: string): string {
  // 运行中 tick:从 created_at 到现在 (秒)。
  const start = new Date(createdAt).getTime()
  const sec = Math.max(0, Math.floor((Date.now() - start) / 1000))
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}m ${s.toString().padStart(2, '0')}s`
}

function statusColor(s: ExecutionTask['status']): string {
  if (s === 'completed') return 'var(--ok)'
  if (s === 'failed') return 'var(--err, #ef4444)'
  if (s === 'cancelled') return 'var(--muted)'
  if (s === 'running') return 'var(--info)'
  return 'var(--warn)' // queued
}

function statusLabel(s: ExecutionTask['status']): string {
  return ({ queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消' } as const)[s]
}

export default function TaskPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: tasks } = useTasks()
  const mode = usePanelStore((s) => s.taskPanelMode)
  const setMode = usePanelStore((s) => s.setTaskPanelMode)
  const [tab, setTab] = useState<'all' | 'done'>('all')
  const [now, setNow] = useState(Date.now())

  // 运行中任务存在 → 1s tick 更新 elapsed 显示。
  useEffect(() => {
    const hasRunning = (tasks ?? []).some((t) => t.status === 'running')
    if (!hasRunning) return
    const h = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(h)
  }, [tasks])

  const running = useMemo(() => (tasks ?? []).filter((t) => RUNNING.has(t.status)), [tasks])
  const done = useMemo(() => (tasks ?? []).filter((t) => TERMINAL.has(t.status)), [tasks])
  const visible = tab === 'all' ? [...running, ...done] : done

  if (!open) return null

  const isDock = mode === 'dock'
  const containerStyle: React.CSSProperties = isDock
    ? {
        position: 'fixed', right: 0, top: 0, bottom: 0, width: 460,
        background: 'var(--bg-accent)', borderLeft: '1px solid var(--border)',
        boxShadow: '-8px 0 24px rgba(0,0,0,0.3)', zIndex: 50,
        display: 'flex', flexDirection: 'column',
      }
    : {
        position: 'fixed', right: 16, bottom: 16, width: 360, height: 480,
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        boxShadow: 'var(--shadow-lg)', borderRadius: 8, zIndex: 50,
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }

  return (
    <>
      <style>{`@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.3 } }`}</style>
      {/* dock 模式才有遮罩;float 不阻塞 canvas */}
      {isDock && (
        <div
          onClick={onClose}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 49 }}
        />
      )}

      <aside style={containerStyle} aria-label="任务面板">
        {/* header */}
        <div style={{
          padding: isDock ? '14px 18px' : '10px 12px',
          borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <h2 style={{
            flex: 1, margin: 0,
            fontSize: isDock ? 14 : 13, fontWeight: 600, color: 'var(--text)',
          }}>
            任务面板
            {running.length > 0 && (
              <span style={{
                marginLeft: 8, padding: '1px 6px', borderRadius: 10,
                background: 'var(--accent-subtle)', color: 'var(--accent)',
                fontSize: 10, fontWeight: 600,
              }}>{running.length} 运行中</span>
            )}
          </h2>
          {/* dock/float toggle */}
          <button
            type="button"
            onClick={() => setMode(isDock ? 'float' : 'dock')}
            aria-label={isDock ? '切换浮窗' : '切换停靠'}
            title={isDock ? '切换浮窗' : '切换停靠'}
            style={iconBtnStyle}
          >
            {isDock ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            style={iconBtnStyle}
          >
            <X size={13} />
          </button>
        </div>

        {/* tab strip */}
        <div style={{
          display: 'flex', gap: 4, padding: '6px 12px',
          borderBottom: '1px solid var(--border)',
        }}>
          <TabBtn active={tab === 'all'} onClick={() => setTab('all')}>全部</TabBtn>
          <TabBtn active={tab === 'done'} onClick={() => setTab('done')}>已完成</TabBtn>
          <span style={{ flex: 1 }} />
          <span style={{ fontSize: 10, color: 'var(--muted)', alignSelf: 'center' }}>
            {visible.length} 项
          </span>
        </div>

        {/* body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: isDock ? 12 : 8 }}>
          {visible.length === 0 && (
            <div style={{
              padding: 32, textAlign: 'center', fontSize: 12, color: 'var(--muted)',
            }}>
              {tab === 'done' ? '暂无完成任务' : '暂无任务 — 跑一个 workflow 试试'}
            </div>
          )}
          {visible.map((t) => (
            t.status === 'running' || t.status === 'queued'
              ? <RunningTaskCard key={t.id} task={t} now={now} />
              : <CompletedTaskCard key={t.id} task={t} />
          ))}
        </div>
      </aside>
    </>
  )
}

const iconBtnStyle: React.CSSProperties = {
  width: 24, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center',
  background: 'transparent', border: 'none', color: 'var(--muted)',
  borderRadius: 4, cursor: 'pointer',
}

function TabBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: '4px 10px', fontSize: 11, fontWeight: 500,
        background: active ? 'var(--accent-subtle)' : 'transparent',
        color: active ? 'var(--accent)' : 'var(--muted)',
        border: 'none', borderRadius: 4, cursor: 'pointer',
      }}
    >{children}</button>
  )
}

function RunningTaskCard({ task, now: _now }: { task: ExecutionTask; now: number }) {
  const cancel = useCancelTask()
  const isCancelled = task.status === 'cancelled'
  const elapsed = tickElapsed(task.created_at)
  const pct = Math.max(0, Math.min(100, task.nodes_total
    ? Math.round((task.nodes_done / task.nodes_total) * 100)
    : 0))
  return (
    <div style={{
      padding: 10, marginBottom: 6,
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 6, display: 'flex', flexDirection: 'column', gap: 6,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {task.status === 'running' ? (
          <Loader2 size={12} className="animate-spin" style={{ color: 'var(--info)' }} />
        ) : (
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: statusColor(task.status), flexShrink: 0,
          }} />
        )}
        <span style={{ flex: 1, fontSize: 12, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {task.workflow_name || `任务 ${String(task.id).slice(0, 8)}`}
        </span>
        <span style={{ fontSize: 10, color: 'var(--muted)' }}>{elapsed}</span>
        {!isCancelled && (
          <button
            type="button"
            onClick={() => cancel.mutate(task.id)}
            disabled={cancel.isPending}
            title="中止"
            aria-label="中止任务"
            style={{ ...iconBtnStyle, color: 'var(--err, #ef4444)' }}
          >
            <XCircle size={14} />
          </button>
        )}
      </div>
      {task.current_node && (
        <div style={{ fontSize: 10, color: 'var(--muted)' }}>
          {task.current_node} · {task.nodes_done}/{task.nodes_total}
        </div>
      )}
      {/* 工作流级进度条(node_done / total) */}
      <div style={{
        height: 3, background: 'var(--border)', borderRadius: 1.5, overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: 'var(--accent)',
          transition: 'width 0.3s linear',
        }} />
      </div>
    </div>
  )
}

function CompletedTaskCard({ task }: { task: ExecutionTask }) {
  const retry = useRetryTask()
  const del = useDeleteTask()
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null)
  const thumb = task.output_thumbnails?.[0]
  const cardRef = useRef<HTMLDivElement>(null)

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    setMenuPos({ x: e.clientX, y: e.clientY })
  }

  const menu: MenuItem[] = [
    ...(thumb ? [{
      label: '查看图片',
      onClick: () => window.open(thumb, '_blank'),
    }] : []),
    {
      label: '复制任务 ID',
      onClick: () => navigator.clipboard.writeText(String(task.id)),
    },
    ...(thumb ? [{
      label: '下载',
      onClick: () => {
        const a = document.createElement('a')
        a.href = thumb
        a.download = `nous-${task.id}.png`
        a.click()
      },
    }] : []),
    { label: '', divider: true } as MenuItem,
    {
      label: '重试',
      onClick: () => retry.mutate(task.id),
      disabled: task.status !== 'failed' && task.status !== 'cancelled',
    },
    {
      label: '删除',
      danger: true,
      onClick: () => del.mutate(task.id),
    },
  ]

  return (
    <>
      <div
        ref={cardRef}
        onContextMenu={onContextMenu}
        style={{
          padding: 8, marginBottom: 4,
          background: 'var(--card)', border: '1px solid var(--border)',
          borderRadius: 6, display: 'flex', alignItems: 'center', gap: 8,
          cursor: 'context-menu',
        }}
      >
        {/* 缩略图 / 占位图标 */}
        <div style={{
          width: 40, height: 40, flexShrink: 0,
          background: 'var(--bg)', border: '1px solid var(--border)',
          borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
          overflow: 'hidden',
        }}>
          {thumb ? (
            <img src={thumb} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          ) : task.task_type === 'image' ? (
            <ImageIcon size={16} style={{ color: 'var(--muted)' }} />
          ) : task.status === 'failed' ? (
            <XCircle size={16} style={{ color: 'var(--err, #ef4444)' }} />
          ) : task.status === 'cancelled' ? (
            <Ban size={16} style={{ color: 'var(--muted)' }} />
          ) : (
            <CheckCircle2 size={16} style={{ color: 'var(--ok)' }} />
          )}
        </div>
        {/* meta */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{
            fontSize: 12, color: 'var(--text)', fontWeight: 500,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {task.workflow_name || `任务 ${String(task.id).slice(0, 8)}`}
          </div>
          <div style={{ fontSize: 10, color: 'var(--muted)', display: 'flex', gap: 6 }}>
            <span style={{ color: statusColor(task.status) }}>● {statusLabel(task.status)}</span>
            <span>{fmtMs(task.duration_ms)}</span>
          </div>
        </div>
        {/* 快捷动作 */}
        <div style={{ display: 'flex', gap: 2 }}>
          {(task.status === 'failed' || task.status === 'cancelled') && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); retry.mutate(task.id) }}
              disabled={retry.isPending}
              title="重试"
              aria-label="重试任务"
              style={iconBtnStyle}
            >
              <RotateCcw size={12} />
            </button>
          )}
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setMenuPos({ x: e.clientX, y: e.clientY }) }}
            title="更多"
            aria-label="更多操作"
            style={iconBtnStyle}
          >
            <ExternalLink size={12} />
          </button>
        </div>
      </div>
      {menuPos && (
        <ContextMenu items={menu} position={menuPos} onClose={() => setMenuPos(null)} />
      )}
    </>
  )
}
