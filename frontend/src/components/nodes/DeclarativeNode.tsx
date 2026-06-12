import { useState, useEffect, useRef, useCallback } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { Zap, Check, ArrowUp, ArrowDown, X, Plus, ImageIcon } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useWorkspaceStore } from '../../stores/workspace'
import { useLightboxStore } from '../../stores/lightbox'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { DECLARATIVE_NODES, type WidgetDef } from '../../models/nodeRegistry'
import { useAgents } from '../../api/agents'
import { apiFetch } from '../../api/client'
import { useEnginesLiveSync, type EngineInfo } from '../../api/engines'
import { useLoras } from '../../api/loras'
import { useComponents, useComponentState, useAllComponentStates, loadedStateByFile, useSeedvr2DitModels, componentStateKey, type ComponentRole, type ComponentLoadState } from '../../api/components'
import BaseNode, { NodeWidgetRow, NodeInput, NodeNumberDrag, NodeTextarea } from './BaseNode'
import NodeSelectPopover from './NodeSelectPopover'
import { readImageDropUrl, isDisplayableImageValue } from './imageDragDrop'

function LoraSelectWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  // V1' Lane C LoadLoRA component node uses this to pick a single LoRA
  // by display name (vs the lora_stack widget which manages an ordered
  // list with strengths for the integrated image_generate node). Source
  // is the same /api/v1/loras scanner endpoint that lora_stack reads —
  // so newly-dropped LoRA files appear in both without an edit.
  const { data: loras } = useLoras()
  const opts = [
    { value: '', label: '— 不应用 LoRA —' },
    ...(loras ?? []).map((lora) => ({ value: lora.name, label: lora.name })),
  ]
  return <NodeSelectPopover value={value} onChange={onChange} options={opts} size="compact" />
}


function AgentSelectWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const { data: agents } = useAgents()
  const opts = (agents ?? []).map((a) => ({ value: a.name, label: a.display_name || a.name }))
  return (
    <NodeSelectPopover
      value={value}
      onChange={onChange}
      options={opts}
      placeholder="选择 Agent..."
      size="compact"
    />
  )
}

interface LoraEntry {
  name: string
  strength: number
}

function LoraStackWidget({
  value,
  onChange,
}: {
  value: LoraEntry[]
  onChange: (v: LoraEntry[]) => void
}) {
  const { data: loras } = useLoras()
  const items = Array.isArray(value) ? value : []

  const update = (next: LoraEntry[]) => onChange(next)
  const add = () => update([...items, { name: '', strength: 1.0 }])
  const remove = (idx: number) => update(items.filter((_, i) => i !== idx))
  const move = (idx: number, dir: -1 | 1) => {
    const target = idx + dir
    if (target < 0 || target >= items.length) return
    const next = items.slice()
    ;[next[idx], next[target]] = [next[target], next[idx]]
    update(next)
  }
  const setName = (idx: number, name: string) => {
    const next = items.slice()
    next[idx] = { ...next[idx], name }
    update(next)
  }
  const setStrength = (idx: number, strength: number) => {
    const next = items.slice()
    next[idx] = { ...next[idx], strength }
    update(next)
  }

  const btnStyle: React.CSSProperties = {
    background: 'var(--bg-hover)',
    border: '1px solid var(--border)',
    borderRadius: 3,
    padding: '2px 4px',
    cursor: 'pointer',
    color: 'var(--muted)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      {items.map((row, idx) => (
        <div key={idx} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <NodeSelectPopover
              value={row.name}
              onChange={(v) => setName(idx, v)}
              options={(loras ?? []).map((l) => ({ value: l.name, label: l.name }))}
              placeholder="选择 LoRA..."
              size="compact"
            />
          </div>
          <div style={{ width: 50 }}>
            <NodeNumberDrag
              value={row.strength}
              onChange={(v) => setStrength(idx, Number(v))}
              min={-2}
              max={2}
              step={0.1}
              precision={2}
            />
          </div>
          <button
            className="nodrag"
            type="button"
            aria-label={`上移 LoRA ${row.name || idx + 1}`}
            onClick={() => move(idx, -1)}
            style={btnStyle}
          >
            <ArrowUp size={10} />
          </button>
          <button
            className="nodrag"
            type="button"
            aria-label={`下移 LoRA ${row.name || idx + 1}`}
            onClick={() => move(idx, 1)}
            style={btnStyle}
          >
            <ArrowDown size={10} />
          </button>
          <button
            className="nodrag"
            type="button"
            aria-label={`删除 LoRA ${row.name || idx + 1}`}
            onClick={() => remove(idx)}
            style={{ ...btnStyle, color: 'var(--err)' }}
          >
            <X size={10} />
          </button>
        </div>
      ))}
      <button
        className="nodrag"
        type="button"
        onClick={add}
        style={{
          ...btnStyle,
          padding: '4px 6px',
          color: 'var(--muted)',
          fontSize: 10,
          gap: 4,
        }}
      >
        <Plus size={10} />
        添加 LoRA
      </button>
    </div>
  )
}

function ModelSelectWidget({
  value,
  onChange,
  filter,
}: {
  value: string
  onChange: (v: string) => void
  filter?: string
}) {
  // Subscribe to /ws/models so the dropdown stays current as models load /
  // unload, even when no other component on the page mounts useEngines().
  useEnginesLiveSync()
  const params = filter ? `?type=${filter}` : ''
  const { data: engines } = useQuery({
    queryKey: ['engines', filter],
    queryFn: () => apiFetch<EngineInfo[]>(`/api/v1/engines${params}`),
  })

  const loaded = (engines ?? []).filter((e) => e.status === 'loaded')
  const unloaded = (engines ?? []).filter((e) => e.status !== 'loaded' && e.local_exists)

  const opts = [
    ...loaded.map((e) => ({
      value: e.name,
      label: e.display_name,
      description: '已加载',
      loaded: true,  // → 绿点 + 「只看已加载」筛选
    })),
    // 未加载:置灰不可选(同旧原生 select 的 disabled 语义)。
    ...unloaded.map((e) => ({
      value: e.name,
      label: e.display_name,
      description: '未加载',
      color: 'var(--muted)',
      disabled: true,
      loaded: false,
    })),
  ]
  return (
    <NodeSelectPopover
      value={value}
      onChange={onChange}
      options={opts}
      placeholder="选择模型..."
      size="compact"
    />
  )
}

export function ComponentSelectWidget({
  value,
  onChange,
  role,
}: { value: string; onChange: (v: string) => void; role: ComponentRole }) {
  const { data: components } = useComponents(role)
  // 已加载状态(按 file 兜底,同 ComponentStatusHeader)→ 下拉标绿点 + 「只看已加载」筛选。
  const { data: allStates } = useAllComponentStates()
  const byFile = loadedStateByFile(allStates)
  const opts = (components ?? []).map((c) => {
    // 同名不同目录的文件(如各模型的 diffusion_pytorch_model.safetensors)在下拉里会看着
    // 一样 —— 副标题附上量化类型 + 末两级目录(模型目录/子目录)区分。
    const parts = (c.abs_path || '').split('/').filter(Boolean)
    const ctx = parts.slice(-3, -1).join('/')
    const quant = c.quant_type && c.quant_type !== 'bf16' ? c.quant_type : ''
    const description = [quant, ctx].filter(Boolean).join(' — ')
    return {
      value: c.abs_path,
      label: c.filename,
      description: description || undefined,
      loaded: byFile[c.abs_path] === 'loaded',
    }
  })
  return (
    <NodeSelectPopover
      value={value}
      onChange={onChange}
      options={opts}
      placeholder={`选择 ${role}...`}
      size="compact"
    />
  )
}

const _STATE_VIS: Record<string, { label: string; color: string }> = {
  loaded:  { label: '已加载', color: 'var(--ok)' },
  loading: { label: '加载中', color: 'var(--warn)' },
  failed:  { label: '失败',   color: 'var(--accent)' },
  cold:    { label: '未加载', color: 'var(--muted)' },
}

export function ComponentStatusHeader({ data }: { data: Record<string, unknown> }) {
  const device = (data.device as string) || 'auto'
  const file = data.file as string | undefined
  const dtype = (data.dtype as string) || 'bfloat16'
  // 显式选卡(cuda:N)用精确 state-key。device=auto 时前端不知道后端把 auto 解析到哪张卡
  // (PR-A 逐组件放置:auto 跟随 transformer 卡,经 get_best_gpu),state-key 里的 'auto'
  // 永远对不上后端注册的 `…|cuda:N|…` → 节点恒显「未加载」(PR-C 修的就是这个)。按 file
  // 兜底匹配(同 #343 service UI loadedStateByFile,robust 于 device/dtype/lora)。
  const exact = useComponentState(componentStateKey({ file, device, dtype }))
  const { data: allStates } = useAllComponentStates()
  const state: ComponentLoadState =
    device === 'auto'
      ? (file ? (loadedStateByFile(allStates)[file] ?? 'cold') : 'cold')
      : exact.state
  const vis = _STATE_VIS[state] ?? _STATE_VIS.cold
  return (
    <div className="flex items-center gap-1.5" style={{ fontSize: 9, color: 'var(--muted)', padding: '2px 10px 4px' }}>
      <span style={{ width: 6, height: 6, borderRadius: 3, background: vis.color, flexShrink: 0 }} />
      <span style={{ color: vis.color }}>{vis.label}</span>
    </div>
  )
}

/** CLIP 节点级聚合四态:多 encoder → 取最差有意义态(任一 failed→failed;任一 loading→
 * loading;全部非空 loaded→loaded;否则 cold)。device 跟随 transformer(auto),按 file 兜底。 */
export function ClipAggregateStatusHeader({ data }: { data: Record<string, unknown> }) {
  const clips = (Array.isArray(data.clips) ? data.clips : []) as { file?: string }[]
  const { data: allStates } = useAllComponentStates()
  const byFile = loadedStateByFile(allStates)
  const files = clips.map((c) => c.file).filter((f): f is string => !!f)
  const states = files.map((f) => byFile[f] ?? 'cold')
  const state: ComponentLoadState =
    files.length === 0 ? 'cold'
      : states.includes('failed') ? 'failed'
        : states.includes('loading') ? 'loading'
          : states.every((s) => s === 'loaded') ? 'loaded'
            : 'cold'
  const vis = _STATE_VIS[state] ?? _STATE_VIS.cold
  return (
    <div className="flex items-center gap-1.5" style={{ fontSize: 9, color: 'var(--muted)', padding: '2px 10px 4px' }}>
      <span style={{ width: 6, height: 6, borderRadius: 3, background: vis.color, flexShrink: 0 }} />
      <span style={{ color: vis.color }}>{vis.label}</span>
    </div>
  )
}

type ClipEntry = { file: string; weight_dtype: string }
const _CLIP_DTYPES = ['default', 'bfloat16', 'fp8_e4m3']

function ClipStateDot({ file }: { file: string }) {
  // device 跟随 transformer(节点级 auto),前端无从得知解析后的卡 → 按 file 兜底匹配
  // 组件状态(同 ComponentStatusHeader 的 auto 分支),否则 per-row 点恒「未加载」。
  const { data: allStates } = useAllComponentStates()
  const state = file ? (loadedStateByFile(allStates)[file] ?? 'cold') : 'cold'
  const vis = _STATE_VIS[state] ?? _STATE_VIS.cold
  return <span title={vis.label} style={{ width: 6, height: 6, borderRadius: 3, background: vis.color, flexShrink: 0 }} />
}

/** PR-3 动态多 CLIP:可增删的 CLIP 编码器列表(每条 file + 精度 + 状态点)。
 * 多编码器执行 gated(runner 拦),但增删 UI + bundle 现在就有。 */
export function ClipStackWidget({
  value,
  onChange,
}: {
  value: ClipEntry[]
  onChange: (v: ClipEntry[]) => void
}) {
  const items = Array.isArray(value) ? value : []
  const add = () => onChange([...items, { file: '', weight_dtype: 'bfloat16' }])
  const remove = (idx: number) => onChange(items.filter((_, i) => i !== idx))
  const setFile = (idx: number, file: string) => {
    const next = items.slice(); next[idx] = { ...next[idx], file }; onChange(next)
  }
  const setDtype = (idx: number, weight_dtype: string) => {
    const next = items.slice(); next[idx] = { ...next[idx], weight_dtype }; onChange(next)
  }
  const btnStyle: React.CSSProperties = {
    background: 'var(--bg-hover)', border: '1px solid var(--border)', borderRadius: 3,
    padding: '2px 4px', cursor: 'pointer', color: 'var(--muted)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      {items.map((row, idx) => (
        <div key={idx} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
          <ClipStateDot file={row.file} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <ComponentSelectWidget value={row.file} onChange={(v) => setFile(idx, v)} role="clip" />
          </div>
          <div style={{ width: 78 }}>
            <NodeSelectPopover
              value={row.weight_dtype || 'bfloat16'}
              onChange={(v) => setDtype(idx, v)}
              options={_CLIP_DTYPES.map((d) => ({ value: d, label: d }))}
              size="compact"
            />
          </div>
          <button
            className="nodrag" type="button"
            aria-label={`删除 CLIP ${idx + 1}`}
            onClick={() => remove(idx)}
            style={{ ...btnStyle, color: 'var(--err)' }}
          >
            <X size={10} />
          </button>
        </div>
      ))}
      <button
        className="nodrag" type="button" onClick={add}
        style={{ ...btnStyle, padding: '4px 6px', fontSize: 10, gap: 4 }}
      >
        <Plus size={10} />
        添加 CLIP
      </button>
    </div>
  )
}

/** clip_stack 取值 + 旧格式兜底:PR-1/PR-2 期存的单 `file` → 包成一条。 */
function clipStackValue(resolved: unknown, nodeData?: Record<string, unknown>): ClipEntry[] {
  if (Array.isArray(resolved) && resolved.length > 0) return resolved as ClipEntry[]
  if (nodeData?.file) {
    return [{ file: String(nodeData.file), weight_dtype: String(nodeData.weight_dtype ?? 'bfloat16') }]
  }
  return Array.isArray(resolved) ? (resolved as ClipEntry[]) : []
}

function resolveValue(value: unknown, widget: WidgetDef): unknown {
  if (value !== undefined && value !== null) return value
  return widget.default
}

function WidgetRenderer({
  widget,
  value,
  onChange,
  nodeData,
}: {
  widget: WidgetDef
  value: unknown
  onChange: (v: unknown) => void
  nodeData?: Record<string, unknown>
}) {
  const resolved = resolveValue(value, widget)

  switch (widget.widget) {
    case 'input':
      return (
        <NodeInput
          value={String(resolved ?? '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={widget.label}
        />
      )
    case 'textarea':
      return (
        <NodeTextarea
          value={String(resolved ?? '')}
          onChange={(e) => onChange(e.target.value)}
          style={widget.rows ? { height: widget.rows * 16 } : undefined}
        />
      )
    case 'select': {
      // options 兼容字符串列表(node.yaml 常写 [default, bfloat16])与对象列表(支持 description/color)。
      const opts = (widget.options ?? []).map((o) =>
        typeof o === 'string' ? { value: o, label: o } : o,
      )
      return (
        <NodeSelectPopover
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
          options={opts}
          size="compact"
        />
      )
    }
    case 'slider':
      return (
        <NodeNumberDrag
          value={Number(resolved ?? widget.min ?? 0)}
          onChange={onChange}
          min={widget.min}
          max={widget.max}
          step={widget.step}
          precision={widget.precision}
        />
      )
    case 'checkbox':
      return (
        <div
          onClick={() => onChange(!resolved)}
          className="nodrag"
          style={{
            width: 32, height: 16, borderRadius: 8, cursor: 'pointer',
            background: resolved ? 'var(--accent)' : 'var(--bg)',
            border: '1px solid var(--border)',
            position: 'relative', transition: 'background 0.2s',
          }}
        >
          <div style={{
            width: 12, height: 12, borderRadius: 6,
            background: '#fff', position: 'absolute', top: 1,
            left: resolved ? 17 : 1, transition: 'left 0.2s',
          }} />
        </div>
      )
    case 'agent_select':
      return (
        <AgentSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    case 'model_select':
      return (
        <ModelSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
          filter={widget.filter}
        />
      )
    case 'lora_stack':
      return (
        <LoraStackWidget
          value={Array.isArray(resolved) ? (resolved as LoraEntry[]) : []}
          onChange={(v) => onChange(v)}
        />
      )
    case 'lora_select':
      return (
        <LoraSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    case 'component_select':
      return (
        <ComponentSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
          role={(widget.role ?? 'diffusion_models') as ComponentRole}
        />
      )
    case 'clip_stack':
      return (
        <ClipStackWidget
          value={clipStackValue(resolved, nodeData)}
          onChange={(v) => onChange(v)}
        />
      )
    case 'image_upload':
      return (
        <ImageUploadWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    case 'seedvr2_model_select':
      return (
        <Seedvr2ModelSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    default:
      return null
  }
}

/** SeedVR2 DiT 模型下拉(混合):白名单全列 —— 盘上有的标「已就绪」(绿点)+ 大小,
 *  其余标「可下载」(灰点,选了 NumZ 从 HF 自动下)。value = filename。 */
function Seedvr2ModelSelectWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const { data: models } = useSeedvr2DitModels()
  const opts = (models ?? []).map((m) => {
    const gb = m.size_mb != null ? ` · ${(m.size_mb / 1024).toFixed(1)}GB` : ''
    return {
      value: m.filename,
      label: m.label,
      description: m.present ? `已就绪${gb} — ${m.desc}` : `可下载(HF)— ${m.desc}`,
      color: m.present ? 'var(--ok)' : 'var(--muted)',
    }
  })
  return (
    <NodeSelectPopover
      value={String(value ?? '')}
      onChange={(v) => onChange(v)}
      options={opts}
      size="compact"
    />
  )
}

/** 图像上传 widget:选/拖/粘贴图 → base64 data URI 存进 node.data。喂 image→image 节点
 *  (SeedVR2 超分等)。有图显示缩略图 + 重传;无图显示上传框。 */
function ImageUploadWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const openLightbox = useLightboxStore((s) => s.openFromUrl)
  const readFile = (file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => onChange((e.target?.result as string) || '')
    reader.readAsDataURL(file)
  }
  // 接受 base64 data URI、本站签名 URL(/files/...)或绝对 http(s) URL。
  // URL 形态来自「出图拖到输入」:画廊缩略图拖进来 / 「转为输入」生成的节点。
  const hasImage = isDisplayableImageValue(value)
  return (
    <div className="nodrag" style={{ width: '100%' }}>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) readFile(f)
        }}
      />
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          // 优先吃拖进来的图片 URL(画廊缩略图 → 输入)。后端 image_input PR-1 已接受本站 URL。
          const url = readImageDropUrl(e.dataTransfer)
          if (url) {
            onChange(url)
            return
          }
          const f = e.dataTransfer.files?.[0]
          if (f) readFile(f)
        }}
        style={{
          width: '100%', minHeight: hasImage ? undefined : 64,
          border: '1px dashed var(--border)', borderRadius: 6, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 6, background: 'var(--bg)', overflow: 'hidden',
        }}
      >
        {hasImage ? (
          <img
            src={value}
            alt="upload"
            title="双击放大预览"
            onDoubleClick={(e) => { e.stopPropagation(); openLightbox(value) }}
            style={{ maxWidth: '100%', maxHeight: 140, borderRadius: 4, display: 'block', cursor: 'zoom-in' }}
          />
        ) : (
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>点击或拖拽上传图片</span>
        )}
      </div>
    </div>
  )
}

export default function DeclarativeNode({ id, type, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const nodeType = type as NodeType
  const declDef = DECLARATIVE_NODES[nodeType]
  const portDef = NODE_DEFS[nodeType]

  // Token stats state
  const [tokenStats, setTokenStats] = useState<{
    phase: 'streaming' | 'done'
    outputTokens: number
    inputTokens: number
    totalTokens: number
    tokensPerSec: number
    durationSec: number
  } | null>(null)
  const tokenCountRef = useRef(0)
  const firstTokenAtRef = useRef<number | null>(null)
  const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // PR-12:**删了旧的 imageStage 3 阶段假进度模拟器**(text_encode 1s →
  // denoise N×1s → vae_decode 0.5s)。它原本挂在 flux2_vae_decode 节点上,
  // 但 VAE Decode 在工作流末尾才执行,前面 Load Diffusion Model / KSampler
  // 跑的时候 VAE 节点根本没 node_start,反过来 VAE 真 node_start 触发后
  // simulation 从「Text encode...」开始,跟实际状态完全不同步 — 用户报告
  // 「正在加载模型,焦点跑到 VAE 那里显示 Denoise」就是这个错配。
  //
  // 现在 backend 已经按真节点 node_id 发 node_progress(KSampler 的 step
  // 事件挂在 KSampler 上),不需要 fake 模拟。每个节点的 denoiseProgress
  // 独立 state,谁收到自己的 step 谁渲染。
  //
  // 节点完成耗时只在「真的跑完」时显示(node_complete);loading / running
  // 的视觉由 BaseNode 通用 status chip(已存在)负责。
  const [doneElapsedSec, setDoneElapsedSec] = useState<number | null>(null)
  // node_complete.cached=true → 该节点结果来自缓存(L1/L2 组件缓存、留噪 latent 等),
  // 没真算 → 完成文案标「(cached)」,跟真跑完区分(见 project_component_l1_cache)。
  const [cachedDone, setCachedDone] = useState(false)
  const nodeStartAtRef = useRef<number | null>(null)
  // PR-3:真采样进度(每步从 backend 经 WS node_progress 事件来)。runner 的
  // P.NodeProgress 用 progress(0-1)+ detail("step N/T")—— parse detail 拿 step/total。
  const [denoiseProgress, setDenoiseProgress] = useState<
    { step: number; total: number; percent: number } | null
  >(null)
  // PR-F:latent 实时 RGB 预览(WS node_progress.preview_url,~96px JPEG data URI)。
  // 「看图慢慢长出来」—— ComfyUI 杀手锏的等价实现。node_complete 清。
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const updateStreamingStats = useCallback(() => {
    const count = tokenCountRef.current
    const first = firstTokenAtRef.current
    if (count < 2 || !first) return
    const elapsed = (performance.now() - first) / 1000
    const rate = elapsed > 0 ? (count - 1) / elapsed : 0
    setTokenStats({
      phase: 'streaming',
      outputTokens: count,
      inputTokens: 0,
      totalTokens: 0,
      tokensPerSec: Math.round(rate * 10) / 10,
      durationSec: Math.round(elapsed * 10) / 10,
    })
  }, [])

  useEffect(() => {
    const handler = (event: CustomEvent) => {
      const data = event.detail
      if (data.type === 'node_stream' && data.node_id === id) {
        tokenCountRef.current++
        if (!firstTokenAtRef.current && tokenCountRef.current === 1) {
          firstTokenAtRef.current = performance.now()
        }
        if (!throttleRef.current) {
          throttleRef.current = setTimeout(() => {
            throttleRef.current = null
            updateStreamingStats()
          }, 250)
        }
      }
      if (data.type === 'node_progress' && data.node_id === id) {
        // 真采样进度:detail = "<stage> N/T"(后端 progress_tracker 发的是
        // "dit_denoise 1/25" / "text_encode 0/25" 等,stage 名前缀随引擎变;早先正则
        // 写死 "step N/T" → 对不上 dit_denoise → 进度条永不显示「RUNNING 无运行进度」)。
        // 放宽:抓任意 "N/T"(可有前缀词),不依赖具体 stage 字面词。progress 字段(0-1)
        // 仍优先用于百分比。
        const m = typeof data.detail === 'string' ? /(\d+)\s*\/\s*(\d+)/.exec(data.detail) : null
        if (m) {
          const step = Number(m[1])
          const total = Number(m[2])
          setDenoiseProgress({
            step,
            total,
            percent: typeof data.progress === 'number' ? Math.round(data.progress * 100) : Math.round((step / total) * 100),
          })
        }
        // PR-F:latent live preview thumbnail(若 backend 发了)。
        if (typeof data.preview_url === 'string' && data.preview_url) {
          setPreviewUrl(data.preview_url)
        }
      }
      if (data.type === 'node_start' && data.node_id === id) {
        // New run on this node — clear previous run's stats
        tokenCountRef.current = 0
        firstTokenAtRef.current = null
        setTokenStats(null)
        setDenoiseProgress(null)
        setPreviewUrl(null)
        setDoneElapsedSec(null)
        setCachedDone(false)
        nodeStartAtRef.current = performance.now()
      }
      if (data.type === 'node_complete' && data.node_id === id) {
        setDenoiseProgress(null)
        setPreviewUrl(null)
        if (throttleRef.current) {
          clearTimeout(throttleRef.current)
          throttleRef.current = null
        }
        const start = nodeStartAtRef.current
        const realElapsed = data.duration_ms
          ? data.duration_ms / 1000
          : start
            ? (performance.now() - start) / 1000
            : 0
        setDoneElapsedSec(realElapsed)
        setCachedDone(!!data.cached)
        nodeStartAtRef.current = null
        const usage = data.usage
        const durationMs = data.duration_ms
        const first = firstTokenAtRef.current
        const elapsed = durationMs
          ? durationMs / 1000
          : first
            ? (performance.now() - first) / 1000
            : 0
        const outTok = usage?.completion_tokens ?? usage?.output_tokens ?? tokenCountRef.current
        const inTok = usage?.prompt_tokens ?? usage?.input_tokens ?? 0
        const total = usage?.total_tokens ?? inTok + outTok
        const rate = elapsed > 0 ? outTok / elapsed : 0
        setTokenStats({
          phase: 'done',
          outputTokens: outTok,
          inputTokens: inTok,
          totalTokens: total,
          tokensPerSec: Math.round(rate * 10) / 10,
          durationSec: Math.round(elapsed * 10) / 10,
        })
        tokenCountRef.current = 0
        firstTokenAtRef.current = null
        // Keep the final stats visible until the next run of this node
        // triggers node_start; no auto-hide timer.
      }
    }
    window.addEventListener('node-progress', handler as any)
    return () => {
      window.removeEventListener('node-progress', handler as any)
      if (throttleRef.current) clearTimeout(throttleRef.current)
    }
  }, [id, nodeType, data.steps, updateStreamingStats])

  if (!declDef || !portDef) return null

  const handleResizeEnd = () => window.dispatchEvent(new Event('node-resize-end'))

  return (
    <>
    <NodeResizer
      isVisible={selected}
      minWidth={220}
      minHeight={80}
      onResizeEnd={handleResizeEnd}
      lineStyle={{ border: 'none' }}
      handleStyle={{ width: 12, height: 12, background: 'transparent', border: 'none' }}
    />
    <BaseNode
      title={declDef.label}
      badge={{
        label: declDef.badge,
        bg: `color-mix(in srgb, ${declDef.badgeColor} 15%, transparent)`,
        color: declDef.badgeColor,
      }}
      selected={selected}
      inputs={portDef.inputs}
      outputs={portDef.outputs}
    >
      {declDef.componentRole === 'clip'
        ? <ClipAggregateStatusHeader data={data as Record<string, unknown>} />
        : declDef.componentRole
          ? <ComponentStatusHeader data={data as Record<string, unknown>} />
          : null}
      {declDef.widgets.map((w) => (
        <NodeWidgetRow key={w.name} label={w.label} stretch={w.widget === 'textarea'}>
          <WidgetRenderer
            widget={w}
            value={data[w.name] as unknown}
            onChange={(v) => updateNode(id, { [w.name]: v })}
            nodeData={data as Record<string, unknown>}
          />
        </NodeWidgetRow>
      ))}
      {/* Streaming text intentionally rendered ONLY in the downstream
          TextOutput node (data flows along edges). LLM node keeps only
          token stats below. */}
      {tokenStats && (
        <div
          className="flex items-center gap-1.5"
          style={{
            fontSize: 9,
            color: 'var(--muted)',
            padding: '4px 10px 6px',
            transition: 'opacity 0.5s',
            opacity: tokenStats.phase === 'done' ? 0.7 : 1,
          }}
        >
          {tokenStats.phase === 'streaming' ? (
            <Zap size={10} style={{ color: 'var(--warn)', flexShrink: 0 }} />
          ) : (
            <Check size={10} style={{ color: 'var(--ok)', flexShrink: 0 }} />
          )}
          {tokenStats.phase === 'streaming' ? (
            <span>
              生成中 · {tokenStats.tokensPerSec} tok/s · 输出 {tokenStats.outputTokens}
            </span>
          ) : (
            <span>
              输入 {tokenStats.inputTokens} · 输出 {tokenStats.outputTokens} · 合计 {tokenStats.totalTokens} · {tokenStats.tokensPerSec} tok/s · {tokenStats.durationSec}s
            </span>
          )}
        </div>
      )}
      {/* PR-F:latent live preview thumbnail(出图过程中节点上叠 96px JPEG,「看图慢慢长出来」)。 */}
      {previewUrl && (
        <div style={{ padding: '4px 10px 0', display: 'flex', justifyContent: 'center' }}>
          <img
            src={previewUrl}
            alt="latent preview"
            style={{
              maxWidth: '100%', maxHeight: 96, borderRadius: 4,
              border: '1px solid var(--border)',
              imageRendering: 'pixelated',
              opacity: 0.95,
            }}
          />
        </div>
      )}
      {/* PR-12:节点级实时进度 + 完成耗时 — **挂在真正跑的那个节点上**,不再
        统一由 VAE Decode 模拟。
        · denoiseProgress 来自 backend 按 node_id 的 node_progress(KSampler 自己发)
        · doneElapsedSec 是任何节点的 node_complete 真实 duration_ms
        · 两者都没有就什么都不渲染,节点保持 BaseNode 默认状态(loading chip 已经
          在 BaseNode 那边显示) */}
      {(denoiseProgress || doneElapsedSec != null) && (
        <div style={{ padding: '4px 10px 6px' }}>
          <div
            className="flex items-center gap-1.5"
            style={{
              fontSize: 9,
              color: 'var(--muted)',
              transition: 'opacity 0.5s',
              opacity: doneElapsedSec != null && !denoiseProgress ? 0.7 : 1,
            }}
          >
            {doneElapsedSec != null && !denoiseProgress ? (
              <Check size={10} style={{ color: 'var(--ok)', flexShrink: 0 }} />
            ) : (
              <ImageIcon size={10} style={{ color: 'var(--info)', flexShrink: 0 }} />
            )}
            <span>
              {denoiseProgress
                ? `step ${denoiseProgress.step}/${denoiseProgress.total} · ${denoiseProgress.percent}%`
                : `完成 · ${Math.round((doneElapsedSec ?? 0) * 10) / 10}s${cachedDone ? ' (cached)' : ''}`}
            </span>
          </div>
          {denoiseProgress && (
            <div style={{
              marginTop: 3, height: 2, background: 'var(--border)', borderRadius: 1, overflow: 'hidden',
            }}>
              <div style={{
                width: `${denoiseProgress.percent}%`, height: '100%',
                background: 'var(--accent)', transition: 'width 0.2s linear',
              }} />
            </div>
          )}
        </div>
      )}
    </BaseNode>
    </>
  )
}
