/**
 * ActiveTaskRow — 任务面板重置 PR-3b:活动任务行(对齐 mockup `.task-row`)。
 *
 *   ┌── timeline-node ──┐   ┌── task-card.active ──────────────────────────┐
 *   │  32×32 圆 + icon  │   │ [IMAGE chip] flux2-klein-bf16  [RUNNING chip]│
 *   │  running glow     │   │ 1024×1024 · seed 42 · cfg 4.5                │
 *   └───────────────────┘   │ ⚡ dit denoise · step 27/50 · 240ms · ETA 5.5s│
 *                           └──────────────────────────────────────────────┘
 *
 * L3 data 来源:useExecutionStore.currentNodeStage/Step/LatencyMs/EtaMs
 * (前端 WS /ws/workflow/{instanceId} 接 PR-1a/b/c/d 后端发的 node_progress 事件)。
 *
 * 多任务并发场景:仅当 task.id == executionStore.taskId(用户刚 Run 的任务)才显示 L3
 * callout;其他 active task 仅显示 status + meta(后端目前不发 global L3 broadcast)。
 *
 * type icon 映射:image → Image、tts → Mic、llm → MessageSquare、vision → Eye。
 */
import { Image as ImageIcon, Mic, MessageSquare, Eye, Zap, Activity } from 'lucide-react'
import type { ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

type TaskType = 'image' | 'tts' | 'vision' | 'llm'

const TYPE_LABEL: Record<TaskType, string> = {
  image: 'IMAGE', tts: 'TTS', vision: 'VISION', llm: 'LLM',
}

const STAGE_LABEL: Record<string, string> = {
  text_encode: 'text encode',
  dit_denoise: 'dit denoise',
  vae_decode: 'vae decode',
  tts_synth: 'tts synth',
  llm_gen: 'llm gen',
  vision_inference: 'vision inference',
}

function getTaskType(t: ExecutionTask): TaskType | null {
  const v = (t as ExecutionTask & { type?: string }).type ?? t.task_type
  return v === 'image' || v === 'tts' || v === 'vision' || v === 'llm' ? v : null
}

function typeIcon(type: TaskType | null) {
  switch (type) {
    case 'image': return ImageIcon
    case 'tts': return Mic
    case 'llm': return MessageSquare
    case 'vision': return Eye
    default: return Activity
  }
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m${Math.round((ms % 60000) / 1000)}s`
}

function getMetaLine(t: ExecutionTask): string {
  // 通用 meta 行 — 优先展示尺寸/seed/cfg(image),其它 service 用 workflow_id 等做兜底。
  const w = t.image_width, h = t.image_height
  const parts: string[] = []
  if (w && h) parts.push(`${w}×${h}`)
  // workflow_id / nodes 信息略,后续 PR-3c 详情卡再展开。
  if (parts.length === 0) {
    parts.push(`#${t.id}`)
  }
  return parts.join(' · ')
}

export default function ActiveTaskRow({ task }: { task: ExecutionTask }) {
  const type = getTaskType(task)
  const Icon = typeIcon(type)
  const isRunning = task.status === 'running'

  const execTaskId = useExecutionStore((s) => s.taskId)
  const stage = useExecutionStore((s) => s.currentNodeStage)
  const step = useExecutionStore((s) => s.currentNodeStep)
  const stepLatencyMs = useExecutionStore((s) => s.currentNodeStepLatencyMs)
  const etaMs = useExecutionStore((s) => s.currentNodeEtaMs)

  // 仅显示用户当前 Run 的 task 的 L3 callout —— 多任务并发时其他行只显示 meta。
  const showL3 = isRunning && execTaskId === task.id && stage != null && step != null

  return (
    <div className="relative flex gap-2.5">
      {/* timeline-node */}
      <div
        className="flex items-center justify-center shrink-0 rounded-full"
        style={{
          width: 32, height: 32,
          border: `2px solid ${isRunning ? 'var(--status-running)' : 'var(--tp-border-strong)'}`,
          background: 'var(--tp-bg-panel)',
          color: isRunning ? 'var(--status-running)' : 'var(--tp-text-muted)',
          boxShadow: isRunning
            ? '0 0 14px rgba(74, 222, 128, 0.55), inset 0 0 8px rgba(74, 222, 128, 0.18)'
            : undefined,
        }}
        aria-label={`task type ${type ?? 'unknown'}`}
      >
        <Icon size={14} strokeWidth={1.8} />
      </div>

      {/* task-card */}
      <div
        className="flex-1 min-w-0 rounded-lg"
        style={{
          padding: '10px 12px',
          background: isRunning
            ? 'linear-gradient(180deg, var(--tp-bg-card) 0%, rgba(74, 222, 128, 0.05) 100%)'
            : 'var(--tp-bg-card)',
          border: `1px solid ${isRunning ? 'rgba(74, 222, 128, 0.4)' : 'var(--tp-border)'}`,
          boxShadow: isRunning
            ? '0 0 0 1px rgba(74, 222, 128, 0.2), 0 6px 16px rgba(74, 222, 128, 0.06)'
            : undefined,
        }}
      >
        {/* task-head */}
        <div className="flex items-center gap-1.5 mb-1.5">
          <div className="flex items-center gap-1.5 flex-1 min-w-0">
            {type && (
              <span
                className="text-[9.5px] font-bold tracking-wider font-mono px-1.5 rounded shrink-0"
                style={{
                  height: 18,
                  display: 'inline-flex',
                  alignItems: 'center',
                  background: `var(--type-${type}-bg-chip)`,
                  color: `var(--type-${type})`,
                }}
              >
                {TYPE_LABEL[type]}
              </span>
            )}
            <span
              className="text-[12.5px] font-semibold font-mono truncate"
              style={{ color: 'var(--tp-text)' }}
              title={task.workflow_name || `#${task.id}`}
            >
              {task.workflow_name || `#${task.id}`}
            </span>
          </div>
          <span
            className="text-[10px] font-semibold tracking-wider uppercase inline-flex items-center gap-1 shrink-0 rounded"
            style={{
              height: 18,
              padding: '0 7px',
              background: isRunning ? 'rgba(74, 222, 128, 0.13)' : 'var(--tp-bg-elevated)',
              color: isRunning ? 'var(--status-running)' : 'var(--tp-text-muted)',
            }}
          >
            {isRunning && (
              <span
                style={{
                  width: 5, height: 5, borderRadius: '50%',
                  background: 'var(--status-running)',
                  boxShadow: '0 0 6px var(--status-running)',
                }}
              />
            )}
            {task.status === 'queued' ? 'queued' : 'running'}
          </span>
        </div>

        {/* task-meta */}
        <div
          className="flex items-center gap-1.5 text-[11px] font-mono"
          style={{ color: 'var(--tp-text-muted)' }}
        >
          {getMetaLine(task).split(' · ').map((seg, i, arr) => (
            <span key={i} className="flex items-center gap-1.5">
              <span>{seg}</span>
              {i < arr.length - 1 && <span style={{ color: 'var(--tp-text-ghost)' }}>·</span>}
            </span>
          ))}
        </div>

        {/* task-callout(L3 progress)*/}
        {showL3 && (
          <div
            className="mt-2 flex items-center gap-1.5 text-[11px] font-mono rounded"
            style={{
              padding: '7px 10px',
              background: 'rgba(74, 222, 128, 0.08)',
              borderLeft: '2px solid var(--status-running)',
              color: 'var(--status-running)',
            }}
          >
            <Zap size={12} strokeWidth={2} />
            <span>
              <span
                className="font-semibold"
                style={{ color: 'var(--status-running-strong, #86efac)' }}
              >
                {STAGE_LABEL[stage!] ?? stage} · step {step!.done}/{step!.total}
              </span>
              {stepLatencyMs != null && stepLatencyMs > 0 && (
                <> · {formatMs(stepLatencyMs)}/step</>
              )}
              {etaMs != null && etaMs > 0 && (
                <> · ETA {formatMs(etaMs)}</>
              )}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
