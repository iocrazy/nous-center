/**
 * TaskMenuButton — Topbar chip:点击 toggle QueueProgressOverlay 展开/收起。
 *
 * v4(2026-05-27 用户截图揭示 popover + overlay 同时存在 bug):删旧 popover 实现。
 *   PR #156 的 popover「暂无运行中任务 / 查看全部」跟 PR #170 浮动 overlay 重复 ——
 *   chip 现在纯入口,点击 toggle 全局 executionStore.taskPanelOpen,
 *   渲染交给 QueueProgressOverlay 自己三态控制(hidden/active/expanded)。
 */
import { useMemo } from 'react'
import { useTasks, type ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

const ACTIVE = new Set<ExecutionTask['status']>(['queued', 'running'])

export default function TaskMenuButton() {
  const { data: tasks } = useTasks()
  const toggle = useExecutionStore((s) => s.toggleTaskPanel)
  const isOpen = useExecutionStore((s) => s.taskPanelOpen)

  const active = useMemo(
    () => (tasks ?? []).filter((t) => ACTIVE.has(t.status)),
    [tasks],
  )
  const isBusy = active.length > 0
  const label = isBusy ? `${active.length} 个活动任务` : '任务'

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      aria-pressed={isOpen}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        fontSize: 12,
        fontWeight: 500,
        background: isOpen ? 'var(--bg-hover)' : isBusy ? 'var(--accent-subtle)' : 'transparent',
        border: '1px solid ' + (isBusy ? 'var(--info)' : 'var(--border)'),
        borderRadius: 6,
        color: isBusy ? 'var(--info)' : 'var(--text)',
        cursor: 'pointer',
        transition: 'background 0.15s, border-color 0.15s',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
      }}
    >
      <span
        style={{
          width: isBusy ? 7 : 6,
          height: isBusy ? 7 : 6,
          borderRadius: '50%',
          background: isBusy ? 'var(--info)' : 'var(--muted)',
          boxShadow: isBusy ? '0 0 6px var(--info)' : 'none',
          flexShrink: 0,
        }}
      />
      <span>{label}</span>
    </button>
  )
}
