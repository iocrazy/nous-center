/**
 * HistoryCard — 任务面板重置 PR-3c:历史任务卡片(对齐 mockup variant-final
 * `.hist-card-expanded` + `.hist-row` 双态)。
 *
 * 两种形态(按 task.id 在 useExecutionStore.expandedHistoryRowIds 决定):
 *
 * 1. Collapsed `.hist-row`(单行 ~50px):
 *      [32×32 mini-thumb] [TYPE chip + name(line 1)
 *                          meta(line 2)]              [time]   [chevron]
 *
 * 2. Expanded `.hist-card-expanded`(随 type 变):
 *      ┌─ exp-head:[TYPE chip] name [DONE chip] [collapse chevron]
 *      ├─ exp-meta:type-specific 参数 · 时间
 *      └─ type-specific body:
 *         · image:64×64 thumb(可点击放大→ PR-3e modal)+ prompt 摘要 + 操作按钮(重跑/复制/下载)
 *         · tts:文字 quote + 简化波形 + 播放按钮
 *         · llm:Q + A 引用 + token stats
 *         · vision:输入图占位 + 描述摘要
 *
 * 默认展开规则:image / tts 默认 expanded(主导视觉占位),llm / vision 默认 collapsed
 * (文本类紧凑)。用户点 chevron 覆盖默认。
 *
 * 数据接口:依赖 ExecutionTask 字段 + 后端 PR-1b/1c/1d 加的 type/audio_duration_seconds/
 * llm_*_tokens/vision_completion_tokens(已在 #175-177 落地)。
 */
import {
  Image as ImageIcon, Mic, MessageSquare, Eye,
  Search, RefreshCw, Copy, Download, Play, ChevronRight,
} from 'lucide-react'
import type { ExecutionTask } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

type TaskType = 'image' | 'tts' | 'vision' | 'llm'

const TYPE_LABEL: Record<TaskType, string> = {
  image: 'IMAGE', tts: 'TTS', vision: 'VISION', llm: 'LLM',
}

const DEFAULT_EXPANDED_TYPES = new Set<TaskType>(['image', 'tts'])

function getTaskType(t: ExecutionTask): TaskType | null {
  const v = (t as ExecutionTask & { type?: string }).type ?? t.task_type
  return v === 'image' || v === 'tts' || v === 'vision' || v === 'llm' ? v : null
}

function getAudioDuration(t: ExecutionTask): number | null {
  return (t as ExecutionTask & { audio_duration_seconds?: number | null }).audio_duration_seconds ?? null
}

function getLlmTokens(t: ExecutionTask): { prompt: number | null; completion: number | null } {
  const o = t as ExecutionTask & {
    llm_prompt_tokens?: number | null
    llm_completion_tokens?: number | null
  }
  return { prompt: o.llm_prompt_tokens ?? null, completion: o.llm_completion_tokens ?? null }
}

function getVisionTokens(t: ExecutionTask): number | null {
  return (t as ExecutionTask & { vision_completion_tokens?: number | null })
    .vision_completion_tokens ?? null
}

/**
 * PR-5:真实 task.result 形态是 `{"outputs": {node_id: envelope}}` —— workflow_executor
 * 包了一层 outputs。本 helper 同时兼容两种 shape:有 outputs 字段时 iter outputs.values();
 * 否则 iter result.values()(旧 fake / 测试用)。
 */
function* iterNodeOutputs(t: ExecutionTask): Iterable<Record<string, unknown>> {
  if (!t.result || typeof t.result !== 'object') return
  const r = t.result as Record<string, unknown>
  const outputs = r.outputs as Record<string, unknown> | undefined
  const source = outputs && typeof outputs === 'object' ? outputs : r
  for (const v of Object.values(source)) {
    if (v && typeof v === 'object') yield v as Record<string, unknown>
  }
}

function getFirstThumbnail(t: ExecutionTask): string | null {
  // 优先 V1.5 Lane I 后端字段 output_thumbnails(若有);
  // 否则 fallback 到 result.outputs.{out}.image_url(真实任务输出 URL)。
  const arr = (t as ExecutionTask & { output_thumbnails?: string[] | null }).output_thumbnails
  if (Array.isArray(arr) && arr.length > 0) return arr[0]
  for (const v of iterNodeOutputs(t)) {
    const imageUrl = v.image_url
    if (typeof imageUrl === 'string' && imageUrl) return imageUrl
  }
  return null
}

function getFirstPrompt(t: ExecutionTask): string | null {
  // 优先找 envelope 上的 prompt 字段(image_generate 输出 / meta.prompt);
  // 兜底找 text_input / encode_prompt 节点的 text 内容(workflow 第一个文本输入)。
  for (const v of iterNodeOutputs(t)) {
    const meta = v.meta as Record<string, unknown> | undefined
    const prompt = (v.prompt ?? meta?.prompt) as string | undefined
    if (typeof prompt === 'string' && prompt) return prompt
  }
  // 兜底:text_input 节点的 text(workflow 起点常用)
  for (const v of iterNodeOutputs(t)) {
    const text = v.text as string | undefined
    // 跳过 LLM response text(LLM result 有 usage 字段),只取纯 text_input
    if (typeof text === 'string' && text && !v.usage) return text
  }
  return null
}

function getLlmText(t: ExecutionTask): string | null {
  for (const v of iterNodeOutputs(t)) {
    // LLM result 一定带 usage,这样区分 text_input / LLM 输出
    if (typeof v.text === 'string' && v.text && v.usage) return v.text
  }
  return null
}

function relativeTime(iso: string | null): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (!Number.isFinite(t)) return ''
  const ms = Date.now() - t
  if (ms < 60_000) return 'just'
  if (ms < 3600_000) return `${Math.floor(ms / 60_000)}m ago`
  if (ms < 86400_000) return `${Math.floor(ms / 3600_000)}h ago`
  return `${Math.floor(ms / 86400_000)}d ago`
}

function durationLabel(ms: number | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

/* ============================================================
 * Main component
 * ============================================================ */
export default function HistoryCard({ task }: { task: ExecutionTask }) {
  const type = getTaskType(task)
  const expandedSet = useExecutionStore((s) => s.expandedHistoryRowIds)
  const toggle = useExecutionStore((s) => s.toggleHistoryRowExpanded)

  // 是否展开:用户显式 toggle 过 → 用 toggle 状态;否则按 type 默认。
  const userToggled = expandedSet.has(task.id)
  const defaultExpanded = type != null && DEFAULT_EXPANDED_TYPES.has(type)
  const expanded = userToggled !== defaultExpanded  // XOR:toggle 翻转 default

  if (!expanded) {
    return <HistoryRowCollapsed task={task} onExpand={() => toggle(task.id)} />
  }
  return <HistoryCardExpanded task={task} onCollapse={() => toggle(task.id)} />
}

/* ============================================================
 * Collapsed row
 * ============================================================ */
function HistoryRowCollapsed({
  task, onExpand,
}: { task: ExecutionTask; onExpand: () => void }) {
  const type = getTaskType(task)
  return (
    <button
      onClick={onExpand}
      className="w-full flex items-center gap-2.5 p-2 rounded transition-colors hover:bg-[var(--tp-bg-hover)] text-left"
      style={{ border: '1px solid transparent', cursor: 'pointer' }}
    >
      <MiniThumb task={task} />
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div className="flex items-center gap-1.5 min-w-0">
          {type && <TypeChip type={type} />}
          <span
            className="text-xs font-mono truncate"
            style={{ color: 'var(--tp-text)' }}
            title={task.workflow_name || `#${task.id}`}
          >
            {task.workflow_name || `#${task.id}`}
          </span>
        </div>
        <div className="text-[10.5px] font-mono" style={{ color: 'var(--tp-text-muted)' }}>
          {summaryMetaLine(task, type)}
        </div>
      </div>
      <span
        className="text-[10px] font-mono shrink-0"
        style={{ color: 'var(--tp-text-faint)' }}
      >
        {relativeTime(task.updated_at ?? task.created_at)}
      </span>
      <ChevronRight size={14} style={{ color: 'var(--tp-text-faint)' }} />
    </button>
  )
}

function summaryMetaLine(task: ExecutionTask, type: TaskType | null): string {
  switch (type) {
    case 'image': {
      const w = task.image_width, h = task.image_height
      return [
        w && h ? `${w}×${h}` : null,
        durationLabel(task.duration_ms),
      ].filter(Boolean).join(' · ')
    }
    case 'tts': {
      const dur = getAudioDuration(task)
      return [
        dur != null ? `${dur.toFixed(1)}s` : null,
        durationLabel(task.duration_ms),
      ].filter(Boolean).join(' · ')
    }
    case 'llm': {
      const { prompt, completion } = getLlmTokens(task)
      const total = (prompt ?? 0) + (completion ?? 0)
      // 速率 tok/s = completion / duration_s,对齐 mockup `234 tok · 18 tok/s · 13s`。
      const tps = completion != null && task.duration_ms && task.duration_ms > 0
        ? Math.round(completion / (task.duration_ms / 1000))
        : null
      return [
        total > 0 ? `${total} tok` : null,
        tps != null ? `${tps} tok/s` : null,
        durationLabel(task.duration_ms),
      ].filter(Boolean).join(' · ')
    }
    case 'vision': {
      const ct = getVisionTokens(task)
      return [
        ct != null ? `${ct} tok` : null,
        durationLabel(task.duration_ms),
      ].filter(Boolean).join(' · ')
    }
    default:
      return durationLabel(task.duration_ms) || task.status
  }
}

/* ============================================================
 * Expanded card
 * ============================================================ */
function HistoryCardExpanded({
  task, onCollapse,
}: { task: ExecutionTask; onCollapse: () => void }) {
  const type = getTaskType(task)
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        background: 'var(--tp-bg-card)',
        border: '1px solid var(--tp-border)',
      }}
    >
      {/* exp-head */}
      <div className="flex items-center gap-1.5 px-3 py-2">
        {type && <TypeChip type={type} />}
        <span
          className="text-[12.5px] font-mono font-semibold truncate flex-1"
          style={{ color: 'var(--tp-text)' }}
          title={task.workflow_name || `#${task.id}`}
        >
          {task.workflow_name || `#${task.id}`}
        </span>
        <StatusChip status={task.status} />
        <button
          onClick={onCollapse}
          aria-label="折叠"
          className="shrink-0 p-0.5"
          style={{ color: 'var(--tp-text-faint)', cursor: 'pointer' }}
        >
          <ChevronRight size={14} style={{ transform: 'rotate(90deg)' }} />
        </button>
      </div>

      {/* type-specific body */}
      {type === 'image' && <ImageExpandedBody task={task} />}
      {type === 'tts' && <TtsExpandedBody task={task} />}
      {type === 'llm' && <LlmExpandedBody task={task} />}
      {type === 'vision' && <VisionExpandedBody task={task} />}
      {!type && <GenericExpandedBody task={task} />}
    </div>
  )
}

/* ============================================================
 * Image body
 * ============================================================ */
function ImageExpandedBody({ task }: { task: ExecutionTask }) {
  const thumb = getFirstThumbnail(task)
  const prompt = getFirstPrompt(task)
  const openModal = useExecutionStore((s) => s.openDetailModal)
  const onOpen = () => openModal(task.id)
  // 从 result.outputs.{ksm}.seed / .meta.seed 找(真实任务里 KSampler 节点带 seed)。
  const seed = (() => {
    for (const v of iterNodeOutputs(task)) {
      if (typeof v.seed === 'number') return v.seed
      const meta = v.meta as Record<string, unknown> | undefined
      if (meta && typeof meta.seed === 'number') return meta.seed
    }
    return null
  })()
  return (
    <>
      <ExpMeta task={task} segments={[
        seed != null ? `seed ${seed}` : null,
        task.image_width && task.image_height ? `${task.image_width}×${task.image_height}` : null,
        durationLabel(task.duration_ms),
      ]} />
      <div className="flex items-center gap-2 px-3 pb-3">
        {/* thumb-64 → 点击打开 detail modal (PR-3e) */}
        <button
          onClick={onOpen}
          className="relative shrink-0 rounded overflow-hidden transition-transform hover:scale-105"
          style={{
            width: 64, height: 64,
            background: thumb ? `center / cover url(${JSON.stringify(thumb)})` : 'var(--tp-bg-elevated)',
            border: '1px solid var(--tp-border-strong)',
            cursor: 'pointer',
          }}
          aria-label="放大查看"
        >
          {!thumb && (
            <div className="w-full h-full flex items-center justify-center">
              <ImageIcon size={24} style={{ color: 'var(--type-image)' }} />
            </div>
          )}
        </button>
        {/* thumb-info → 点击同样打开 modal(扩大点击区) */}
        <button
          onClick={onOpen}
          className="flex-1 min-w-0 flex flex-col gap-1 text-left"
          style={{ cursor: 'pointer', background: 'transparent', border: 'none', padding: 0 }}
          aria-label="放大查看(prompt)"
        >
          <div
            className="flex items-center gap-1.5 text-[11px] font-mono"
            style={{ color: 'var(--type-image)' }}
          >
            <Search size={12} />
            <span>点击放大 →</span>
          </div>
          <div
            className="text-[11px] font-mono truncate"
            style={{ color: 'var(--tp-text-muted)' }}
            title={prompt ?? ''}
          >
            {prompt ? `prompt: ${prompt}` : <span style={{ color: 'var(--tp-text-faint)' }}>(no prompt)</span>}
          </div>
        </button>
        {/* thumb-actions */}
        <div className="flex gap-1 shrink-0">
          <IconBtn label="重跑" primary><RefreshCw size={12} /></IconBtn>
          <IconBtn label="复制"><Copy size={12} /></IconBtn>
          <IconBtn label="下载"><Download size={12} /></IconBtn>
        </div>
      </div>
    </>
  )
}

/* ============================================================
 * TTS body
 * ============================================================ */
function TtsExpandedBody({ task }: { task: ExecutionTask }) {
  const dur = getAudioDuration(task)
  // 计算 RT(realtime factor)= audio_duration / synth_duration,对齐 mockup
  // `15.8s · 22.05kHz · 1×RT`。<1 = 实时,>1 = 比实时快。
  const rt = dur != null && task.duration_ms && task.duration_ms > 0
    ? (dur / (task.duration_ms / 1000))
    : null
  return (
    <>
      <ExpMeta task={task} segments={[
        dur != null ? `${dur.toFixed(1)}s` : null,
        rt != null ? `${rt.toFixed(1)}×RT` : null,
        durationLabel(task.duration_ms),
      ]} />
      {/* tts-text quote */}
      <div
        className="mx-3 mb-2 p-2 rounded text-[11.5px] leading-relaxed"
        style={{
          background: 'var(--type-tts-quote-bg, #0b1518)',
          border: '1px solid var(--type-tts-border-subtle, rgba(34, 211, 238, 0.25))',
          color: 'var(--tp-text)',
        }}
      >
        <span
          className="text-base mr-1 font-serif"
          style={{ color: 'var(--type-tts)' }}
        >&ldquo;</span>
        <span className="font-mono">{getFirstPrompt(task) ?? '(audio output)'}</span>
        <span
          className="text-base ml-1 font-serif"
          style={{ color: 'var(--type-tts)' }}
        >&rdquo;</span>
      </div>
      {/* audio-player */}
      <div
        className="mx-3 mb-3 flex items-center gap-2 p-2 rounded"
        style={{
          background: 'var(--type-tts-audio-bg, #0e1a1d)',
          border: '1px solid var(--type-tts-audio-border, #1a3038)',
        }}
      >
        <button
          aria-label="播放"
          className="shrink-0 flex items-center justify-center rounded-full"
          style={{
            width: 26, height: 26,
            background: 'var(--type-tts)',
            color: '#0a0a0c',
            cursor: 'pointer',
          }}
        >
          <Play size={11} strokeWidth={2} fill="currentColor" />
        </button>
        <div className="flex-1 flex items-end gap-px h-5 overflow-hidden">
          {/* 模拟波形:30 个高度参数化 bar */}
          {Array.from({ length: 30 }, (_, i) => i).map((i) => {
            const h = 4 + ((i * 37 + 11) % 16)
            const played = i < 11
            return (
              <span
                key={i}
                style={{
                  width: 2,
                  height: h,
                  background: played ? 'var(--type-tts)' : 'var(--tp-text-faint)',
                  opacity: played ? 1 : 0.5,
                }}
              />
            )
          })}
        </div>
        <span
          className="shrink-0 text-[10px] font-mono"
          style={{ color: 'var(--tp-text-muted)' }}
        >
          {dur != null ? `0:00 / ${dur.toFixed(1)}s` : '0:00'}
        </span>
      </div>
    </>
  )
}

/* ============================================================
 * LLM body
 * ============================================================ */
function LlmExpandedBody({ task }: { task: ExecutionTask }) {
  const { prompt, completion } = getLlmTokens(task)
  const text = getLlmText(task)
  return (
    <>
      <ExpMeta task={task} segments={[
        prompt != null && completion != null ? `${prompt + completion} tok` : null,
        prompt != null ? `${prompt} prompt` : null,
        completion != null ? `${completion} gen` : null,
        durationLabel(task.duration_ms),
      ]} />
      <div
        className="mx-3 mb-3 p-2.5 rounded text-[11.5px] leading-relaxed font-mono"
        style={{
          background: 'var(--type-llm-bg-history, rgba(96, 165, 250, 0.08))',
          border: '1px solid var(--type-llm-border-history, rgba(96, 165, 250, 0.35))',
          borderLeft: '3px solid var(--type-llm)',
          color: 'var(--tp-text)',
          maxHeight: 80,
          overflow: 'hidden',
        }}
      >
        <span style={{ color: 'var(--type-llm)', fontWeight: 600 }}>A: </span>
        {text ? (
          <span>{text.length > 200 ? text.slice(0, 200) + '…' : text}</span>
        ) : (
          <span style={{ color: 'var(--tp-text-faint)' }}>(no text)</span>
        )}
      </div>
    </>
  )
}

/* ============================================================
 * Vision body
 * ============================================================ */
function VisionExpandedBody({ task }: { task: ExecutionTask }) {
  const ct = getVisionTokens(task)
  const text = getLlmText(task)
  return (
    <>
      <ExpMeta task={task} segments={[
        ct != null ? `${ct} tok` : null,
        durationLabel(task.duration_ms),
      ]} />
      <div className="px-3 pb-3 flex items-start gap-2">
        <div
          className="shrink-0 flex items-center justify-center rounded"
          style={{
            width: 56, height: 56,
            background: 'var(--type-vision-bg-card, #1a0f06)',
            border: '1px solid var(--type-vision-border-subtle, rgba(251, 146, 60, 0.25))',
            color: 'var(--type-vision)',
          }}
          aria-label="输入图占位"
        >
          <Eye size={20} />
        </div>
        <div
          className="flex-1 text-[11.5px] leading-relaxed font-mono"
          style={{ color: 'var(--tp-text)' }}
        >
          {text ? (
            <span>{text.length > 160 ? text.slice(0, 160) + '…' : text}</span>
          ) : (
            <span style={{ color: 'var(--tp-text-faint)' }}>(no description)</span>
          )}
        </div>
      </div>
    </>
  )
}

function GenericExpandedBody({ task }: { task: ExecutionTask }) {
  return (
    <ExpMeta task={task} segments={[task.status, durationLabel(task.duration_ms)]} />
  )
}

/* ============================================================
 * Subs
 * ============================================================ */
function ExpMeta({
  task, segments,
}: {
  task: ExecutionTask
  segments: (string | null)[]
}) {
  const visible = segments.filter(Boolean) as string[]
  const time = relativeTime(task.updated_at ?? task.created_at)
  return (
    <div className="flex items-center gap-1.5 px-3 pb-2 text-[11px] font-mono"
      style={{ color: 'var(--tp-text-muted)' }}>
      {visible.map((s, i) => (
        <span key={i} className="flex items-center gap-1.5">
          <span>{s}</span>
          {(i < visible.length - 1 || time) && (
            <span style={{ color: 'var(--tp-text-ghost)' }}>·</span>
          )}
        </span>
      ))}
      <span className="ml-auto" style={{ color: 'var(--tp-text-faint)' }}>{time}</span>
    </div>
  )
}

function TypeChip({ type }: { type: TaskType }) {
  return (
    <span
      className="text-[9.5px] font-bold tracking-wider font-mono px-1.5 rounded shrink-0 inline-flex items-center"
      style={{
        height: 18,
        background: `var(--type-${type}-bg-chip)`,
        color: `var(--type-${type})`,
      }}
    >
      {TYPE_LABEL[type]}
    </span>
  )
}

function StatusChip({ status }: { status: ExecutionTask['status'] }) {
  const isDone = status === 'completed'
  return (
    <span
      className="text-[10px] font-semibold tracking-wider uppercase shrink-0 rounded inline-flex items-center"
      style={{
        height: 18,
        padding: '0 7px',
        background: isDone ? 'var(--tp-bg-elevated)' : 'rgba(255, 159, 10, 0.13)',
        color: isDone ? 'var(--tp-text-muted)' : 'var(--status-failed, #f87171)',
      }}
    >
      {status === 'completed' ? 'done'
       : status === 'failed' ? 'failed'
       : status === 'cancelled' ? 'cancelled'
       : status}
    </span>
  )
}

function IconBtn({
  children, label, primary, onClick,
}: {
  children: React.ReactNode
  label: string
  primary?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className="flex items-center justify-center rounded transition-colors"
      style={{
        width: 26, height: 26,
        background: primary ? 'var(--type-image-bg-chip)' : 'var(--tp-bg-elevated)',
        border: `1px solid ${primary ? 'var(--type-image-border-subtle, var(--type-image))' : 'var(--tp-border-strong)'}`,
        color: primary ? 'var(--type-image)' : 'var(--tp-text-muted)',
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

function MiniThumb({ task }: { task: ExecutionTask }) {
  const type = getTaskType(task)
  const thumb = type === 'image' ? getFirstThumbnail(task) : null
  if (thumb) {
    return (
      <div
        className="shrink-0 rounded"
        style={{
          width: 32, height: 32,
          background: `center / cover url(${JSON.stringify(thumb)})`,
          border: '1px solid var(--tp-border-strong)',
        }}
      />
    )
  }
  // 无 thumb → 用 type icon + 类型色背景
  return (
    <div
      className="shrink-0 rounded flex items-center justify-center"
      style={{
        width: 32, height: 32,
        background: type ? `var(--type-${type}-bg-chip)` : 'var(--tp-bg-elevated)',
        border: `1px solid var(--tp-border-strong)`,
        color: type ? `var(--type-${type})` : 'var(--tp-text-muted)',
      }}
    >
      {type === 'image' && <ImageIcon size={14} />}
      {type === 'tts' && <Mic size={14} />}
      {type === 'llm' && <MessageSquare size={14} />}
      {type === 'vision' && <Eye size={14} />}
    </div>
  )
}
