/** TaskPanel 排序纯函数 + 标签常量。
 *
 * 独立文件原因:TaskPanel.tsx 是组件文件,react-refresh/only-export-components
 * 不允许混合组件 + 工具导出。抽到 .ts 文件保 fast-refresh 干净。
 */
import type { ExecutionTask } from '../../api/tasks'
import type { TaskSortDir, TaskSortKey, TaskStatus } from '../../stores/panel'

export const STATUS_LABEL: Readonly<Record<TaskStatus, string>> = {
  queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消',
}

export const SORT_LABEL: Readonly<Record<`${TaskSortKey}.${TaskSortDir}`, string>> = {
  'created.desc': '创建时间(新→旧)',
  'created.asc': '创建时间(旧→新)',
  'duration.desc': '耗时(长→短)',
  'duration.asc': '耗时(短→长)',
}

/** 按相对日期分组(对齐 ComfyUI「昨天/今天/上周」grouping)。
 *  返回 [(label, items), ...],label 是 「今天」/「昨天」/「本周」/「上周」/「更早」。
 *  按时间倒序保持(最新分组在前),items 内顺序由 caller 保留(已 sort 过)。 */
export function groupByDate(tasks: readonly ExecutionTask[]): Array<[string, ExecutionTask[]]> {
  const now = new Date()
  const today0 = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const yesterday0 = today0 - 86_400_000
  const dow = now.getDay() === 0 ? 7 : now.getDay()
  const weekStart0 = today0 - (dow - 1) * 86_400_000
  const lastWeekStart0 = weekStart0 - 7 * 86_400_000

  const buckets: Record<string, ExecutionTask[]> = {
    今天: [], 昨天: [], 本周: [], 上周: [], 更早: [],
  }
  for (const t of tasks) {
    const ts = new Date(t.created_at).getTime()
    if (ts >= today0) buckets['今天'].push(t)
    else if (ts >= yesterday0) buckets['昨天'].push(t)
    else if (ts >= weekStart0) buckets['本周'].push(t)
    else if (ts >= lastWeekStart0) buckets['上周'].push(t)
    else buckets['更早'].push(t)
  }
  return (['今天', '昨天', '本周', '上周', '更早'] as const)
    .map((k) => [k, buckets[k]] as [string, ExecutionTask[]])
    .filter(([, items]) => items.length > 0)
}


/** 排序 visible 列表。
 * - created:用 created_at ISO 串字典序,与时间序一致;
 * - duration:running/queued 没 duration_ms 视为 +Infinity 排末尾(无论升降都在尾),
 *   保证「耗时」排序看到的都是完成态任务。 */
export function sortTasks(
  tasks: readonly ExecutionTask[],
  key: TaskSortKey,
  dir: TaskSortDir,
): ExecutionTask[] {
  const sign = dir === 'asc' ? 1 : -1
  const arr = [...tasks]
  if (key === 'created') {
    arr.sort((a, b) => sign * a.created_at.localeCompare(b.created_at))
  } else {
    arr.sort((a, b) => {
      const da = a.duration_ms ?? Number.POSITIVE_INFINITY
      const db = b.duration_ms ?? Number.POSITIVE_INFINITY
      // 把没 duration 的任务永远排到末尾,不被 desc/asc 翻动。
      if (da === Number.POSITIVE_INFINITY && db !== Number.POSITIVE_INFINITY) return 1
      if (db === Number.POSITIVE_INFINITY && da !== Number.POSITIVE_INFINITY) return -1
      return sign * (da - db)
    })
  }
  return arr
}
