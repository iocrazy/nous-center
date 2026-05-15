import { useEffect, useRef } from 'react'
import { useTasks, type ExecutionTask } from '../api/tasks'
import { useNotificationStore } from '../stores/notifications'
import { useToastStore } from '../stores/toast'

const TERMINAL: ReadonlySet<ExecutionTask['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
])

function isTerminal(status: ExecutionTask['status']): boolean {
  return TERMINAL.has(status)
}

function fmtDuration(ms: number | null): string {
  if (ms == null || ms <= 0) return ''
  const s = ms / 1000
  if (s < 60) return ` · ${s.toFixed(0)}s`
  const m = Math.floor(s / 60)
  return ` · ${m}m${Math.round(s - m * 60)}s`
}

/** 全局挂载一次（App.tsx）。监听 useTasks() 数据，某 task 从非终态翻到
 * 终态时发一次完成/失败通知。spec §6.3 DD6。
 *
 * 检测靠「上一帧 status」与「这一帧 status」的 diff —— 不解析 WS 事件类型，
 * 所以不和 Lane G 的 /ws/tasks 推送实现耦合。初次加载时已是终态的 task
 * 不触发（prevStatus 为空 = 不算「转换」）。 */
export function useTaskCompletionNotifier(): void {
  const { data: tasks } = useTasks()
  const { notifyOnce } = useNotificationStore()
  const toast = useToastStore((s) => s.add)
  // task_id → 上一帧观测到的 status。
  const prevStatus = useRef<Map<string, ExecutionTask['status']>>(new Map())

  useEffect(() => {
    if (!tasks) return
    const seen = prevStatus.current
    for (const t of tasks) {
      const before = seen.get(t.id)
      const justFinished =
        before !== undefined && !isTerminal(before) && isTerminal(t.status)
      if (justFinished) {
        if (t.status === 'completed') {
          notifyOnce(t.id, `${t.workflow_name} 完成${fmtDuration(t.duration_ms)}`, 'success', toast)
        } else if (t.status === 'failed') {
          notifyOnce(t.id, `${t.workflow_name} 失败`, 'error', toast)
        }
        // cancelled 不发通知（用户自己点的取消，无需打扰）。
      }
      seen.set(t.id, t.status)
    }
  }, [tasks, notifyOnce, toast])
}
