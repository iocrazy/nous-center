import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

/** runner 泳道当前正在跑的任务（spec §6.1 进度条 + step 文案的数据源）。 */
export interface RunnerCurrentTask {
  task_id: string
  workflow_name: string
  /** 0.0 ~ 1.0 —— 泳道进度条宽度。 */
  progress: number
  /** "step 18/30" 之类的细节文案，可空。 */
  detail: string | null
}

/** runner 排队列表里的一条（spec §6.4 排队展开的有序列表项）。 */
export interface RunnerQueueItem {
  task_id: string
  workflow_name: string
  /** 1-based 排队序号，spec §6.4 的 #1 #2 #3。 */
  position: number
}

/** 一个 GPU runner 泳道的完整状态（spec §6.1 / §6.2）。
 *
 * state 与 spec §6.2 异常态表一一对应：
 *   idle    — 灰点 + 「idle」
 *   busy    — 绿点 + current_task 名 + 进度条
 *   restarting — 黄色脉冲点 + 「重启中 N/M」（restart_attempt 提供 N/M）
 *   load_failed — 红点 + 「加载失败: ...」（load_error 提供文案）+ Retry 按钮
 */
export interface RunnerInfo {
  id: string
  /** 展示名，如 "Runner-I"。 */
  label: string
  role: 'image' | 'tts' | 'llm'
  state: 'idle' | 'busy' | 'restarting' | 'load_failed'
  current_task: RunnerCurrentTask | null
  queue: RunnerQueueItem[]
  /** restarting 态：[当前第几次, 总 backoff 次数]，如 [2, 4] → 「重启中 2/4」。 */
  restart_attempt: [number, number] | null
  /** load_failed 态：失败文案，如 "qwen3-35b OOM"。 */
  load_error: string | null
}

/** 拉 runner 泳道数据。
 *
 * 后端 /api/v1/runners 由 V1.5 Lane G/H 提供（RunnerSupervisor 调度态）。
 * 端点尚未落地时 hook 降级为空数组 —— TaskPanel 泳道区显示「暂无 runner 数据」，
 * 不阻塞 Lane I 独立 merge。
 */
export function useRunners() {
  return useQuery<RunnerInfo[]>({
    queryKey: ['runners'],
    queryFn: async () => {
      try {
        return await apiFetch<RunnerInfo[]>('/api/v1/runners')
      } catch (e) {
        // 端点未落地（404）/ 暂时不可达 → 降级空泳道，不让整个面板崩。
        if ((e as { status?: number }).status === 404) return []
        throw e
      }
    },
    // runner 状态变化频繁（进度条、排队数），3s 轮询；WS 推送由后续 Lane 接。
    refetchInterval: 3_000,
  })
}
