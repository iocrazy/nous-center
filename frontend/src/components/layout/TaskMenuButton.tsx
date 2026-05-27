/**
 * TaskMenuButton — Topbar 全局任务入口(对齐 ComfyUI 截图 1:1)。
 *
 * v2 重做(2026-05-27 用户对照 ComfyUI 截图反馈):**ComfyUI 没有 popover 这一层**
 *   chip 点击 = 直接 toggle drawer(TaskPanel),不再分 popover/drawer 两层。
 *   原 popover「当前任务 + live preview + 双进度条」搬到 TaskPanel 顶部(active 任务卡)。
 *
 * chip 文案:`N 个活动任务`(永远显数字,active=0 也显 `0 个活动任务`,对齐 ComfyUI)。
 */
import { useMemo } from 'react'
import { useTasks, type ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

const ACTIVE = new Set<ExecutionTask['status']>(['queued', 'running'])

export default function TaskMenuButton() {
  const { data: tasks } = useTasks()
  const toggleTaskPanel = useExecutionStore((s) => s.toggleTaskPanel)

  const active = useMemo(() => (tasks ?? []).filter((t) => ACTIVE.has(t.status)), [tasks])
  const isBusy = active.length > 0

  return (
    <button
      type="button"
      onClick={toggleTaskPanel}
      aria-label="打开任务面板"
      title={`${active.length} 个活动任务`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        fontSize: 11,
        fontWeight: 500,
        background: isBusy ? 'var(--accent-subtle)' : 'transparent',
        border: '1px solid ' + (isBusy ? 'var(--info)' : 'var(--border)'),
        borderRadius: 6,
        color: isBusy ? 'var(--info)' : 'var(--text)',
        cursor: 'pointer',
        transition: 'background 0.15s, border-color 0.15s',
      }}
    >
      <span
        style={{
          width: isBusy ? 7 : 6, height: isBusy ? 7 : 6, borderRadius: '50%',
          background: isBusy ? 'var(--info)' : 'var(--muted)',
          boxShadow: isBusy ? '0 0 6px var(--info)' : 'none',
          flexShrink: 0,
        }}
      />
      <span>{active.length} 个活动任务</span>
    </button>
  )
}
