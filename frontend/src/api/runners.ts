import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
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
  /** 该 runner 占用的 GPU index 列表（对齐 hardware.yaml groups[].gpus）。
   * Dashboard GpuPanel 用它给每张 GPU 标「属于哪个 runner」（spec §6 DD3）。 */
  gpus: number[]
}

/** 后端 /api/v1/monitor/runners 返回的 snapshot 形状（Lane H Task 6 + Lane K
 * LLMRunner.health_snapshot + Lane K follow-up RunnerClient.current_dispatch）。
 *
 * current_task: RunnerClient 暴露的 inflight dispatch —— image/tts runner 真在
 * 跑节点时非空；LLM runner 走 vLLM HTTP 直连不汇集到此处,恒为 null。
 */
interface BackendRunnerSnapshot {
  group_id: 'image' | 'tts' | 'llm'
  gpus: number[]
  running: boolean
  restart_count: number
  pid: number | null
  current_task: {
    task_id: number
    workflow_name: string
    node_id: string
    node_type: string
    started_at: number
    progress: number   // 0.0 ~ 1.0
    detail: string | null
  } | null
}

interface BackendRunnersResponse {
  runners: BackendRunnerSnapshot[]
}

/** 把后端 snapshot 适配成 UI 期望的 RunnerInfo。
 *
 * 语义对齐:
 *   后端 `running` = 进程存活(spawned + healthy),与「在跑任务」无关。
 *   后端 `current_task` = RunnerClient 跟踪的 inflight dispatch,有 = 真在跑。
 *
 * State 决策:
 *   - !alive + restart_count>0 → 'restarting'(挂了在重启,带 restart_attempt 元组)
 *   - alive + current_task 非空 → 'busy'(spec §6.2 绿点 + 当前任务 + 进度条)
 *   - 其它 → 'idle'(存活待命 / 未启动 / LLM 未 preload)
 *
 * label: group_id 首字母大写,即 "Image" / "Llm" / "Tts"
 */
function adaptBackendSnapshot(s: BackendRunnerSnapshot): RunnerInfo {
  const restarting = s.restart_count > 0 && !s.running
  const current = s.current_task && s.running
    ? {
        task_id: String(s.current_task.task_id),
        workflow_name: s.current_task.workflow_name || s.current_task.node_type,
        progress: s.current_task.progress,
        detail: s.current_task.detail,
      }
    : null
  let state: RunnerInfo['state'] = 'idle'
  if (restarting) state = 'restarting'
  else if (current) state = 'busy'
  return {
    id: s.group_id,
    label: s.group_id.charAt(0).toUpperCase() + s.group_id.slice(1),
    role: s.group_id,
    state,
    current_task: current,
    queue: [],
    restart_attempt: restarting ? [s.restart_count, 4] : null,
    load_error: null,
    gpus: s.gpus,
  }
}

/** 拉 runner 泳道数据。
 *
 * 后端 /api/v1/monitor/runners 由 V1.5 Lane H Task 6 + Lane K 提供
 * (RunnerSupervisor + LLMRunner 的 health_snapshot 合并)。
 * 端点不可达时 hook 降级为空数组 —— TaskPanel 泳道区显示「暂无 runner 数据」。
 */
export function useRunners() {
  return useQuery<RunnerInfo[]>({
    queryKey: ['runners'],
    queryFn: async () => {
      try {
        const resp = await apiFetch<BackendRunnersResponse>('/api/v1/monitor/runners')
        return (resp.runners ?? []).map(adaptBackendSnapshot)
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

/** 触发某 runner 重新加载失败的模型（spec §6.2 DD5 的 Retry 按钮）。
 * 后端 POST /api/v1/runners/{id}/retry 由 Lane H 提供。 */
export function useRetryRunner() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runnerId: string) =>
      apiFetch(`/api/v1/runners/${runnerId}/retry`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runners'] }),
  })
}
