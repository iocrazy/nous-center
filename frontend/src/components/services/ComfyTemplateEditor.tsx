// ComfyUI 模板服务的「编辑」分支(Task 9,spec §「读图选节点配置」)。替代
// comfy_template 来源服务在 ServiceDetail 里的旧节点图编辑器:读原始 ComfyUI 工作流
// JSON,不依赖 nous 声明式节点表(那套只认自家节点类型)。流程:
//   1) sidecar 状态行(在线才能拿 object_info 类型预填,离线仍可手工配置)。
//   2) 重新上传替换 workflow_json(节点结构变了 → 旧映射按 comfy_node_id/comfy_input
//      定位失效,标红提示,不自动删,留给管理员确认)。
//   3) React Flow 通用节点卡(标题=class_type,与 nous 节点图共用视觉语言但节点内容
//      来自原始 class_type/inputs,不经 DECLARATIVE_NODES)。点卡开弹窗配置该节点的
//      原始 inputs(节点引用连线 [nodeId,slot] 不算可配置字段,过滤掉)。
//   4) 弹窗与下方字段汇总表共享同一份 exposedParams state,保存走 putMapping。
import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Upload, X } from 'lucide-react'
import {
  getComfyHealth,
  getComfyTemplate,
  getObjectInfo,
  putMapping,
  reuploadComfyTemplate,
  type ComfyExposedParam,
  type ComfyHealth,
  type ComfyTemplateDetail,
} from '../../api/comfyTemplates'
import { isComfyNodeRef, layoutComfyGraph, type ComfyWorkflow } from './comfyGraphLayout'
import ComfyTemplateGraph from './ComfyTemplateGraph'

/** 弹窗/汇总表共享的单行草稿。与后端 mapping 契约(comfy_node_id/comfy_input)同形 —
 *  这里就是 ComfyExposedParam 的别名,不额外分叉一套类型跟后端契约慢慢漂移。 */
export type ExposedParamDraft = ComfyExposedParam

export interface ComfyTemplateEditorProps {
  templateId: string | number
}

interface RawInputRow {
  inputName: string
  rawValue: unknown
}

interface FieldMeta {
  type: string
  min?: number
  max?: number
  step?: number
  options?: unknown[]
}

const NUMERIC_TYPES = new Set(['integer', 'number'])
const TYPE_OPTIONS: Array<{ v: string; label: string }> = [
  { v: 'string', label: '文本' },
  { v: 'integer', label: '整数' },
  { v: 'number', label: '小数' },
  { v: 'boolean', label: '开关' },
]

function classInputDecl(
  classType: string,
  inputName: string,
  objectInfo: Record<string, unknown> | null,
): [unknown, Record<string, unknown>?] | null {
  const cls = objectInfo?.[classType] as
    | { input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> } }
    | undefined
  const input = cls?.input
  const decl = input?.required?.[inputName] ?? input?.optional?.[inputName]
  return Array.isArray(decl) ? (decl as [unknown, Record<string, unknown>?]) : null
}

function numericConstraints(c?: Record<string, unknown>): Partial<FieldMeta> {
  if (!c) return {}
  const out: Partial<FieldMeta> = {}
  if (typeof c.min === 'number') out.min = c.min
  if (typeof c.max === 'number') out.max = c.max
  if (typeof c.step === 'number') out.step = c.step
  return out
}

/** object_info 有该 class 的该 input 声明 → 按它给的 TYPE/OPTIONS + min/max/step 走;
 *  否则(sidecar 离线、或该节点类型未在 object_info 出现)按工作流里的原始值类型兜底推断。 */
function deriveFieldMeta(
  classType: string,
  inputName: string,
  rawValue: unknown,
  objectInfo: Record<string, unknown> | null,
): FieldMeta {
  const decl = classInputDecl(classType, inputName, objectInfo)
  if (decl) {
    const [typeOrOptions, constraints] = decl
    if (Array.isArray(typeOrOptions)) {
      return { type: 'string', options: typeOrOptions, ...numericConstraints(constraints) }
    }
    const t = String(typeOrOptions).toUpperCase()
    const type = t === 'INT' ? 'integer' : t === 'FLOAT' ? 'number' : t === 'BOOLEAN' ? 'boolean' : 'string'
    return { type, ...numericConstraints(constraints) }
  }
  if (typeof rawValue === 'boolean') return { type: 'boolean' }
  if (typeof rawValue === 'number') return { type: Number.isInteger(rawValue) ? 'integer' : 'number' }
  return { type: 'string' }
}

function isSeedLike(inputName: string, type: string): boolean {
  return type === 'integer' && /seed/i.test(inputName)
}

/** 同 ImportComfyDialog 的导入格式校验(API 格式 = 每个顶层值都是带 class_type 的节点对象)。 */
function isApiFormatWorkflow(parsed: unknown): parsed is ComfyWorkflow {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return false
  const values = Object.values(parsed as Record<string, unknown>)
  if (values.length === 0) return false
  return values.every(
    (v) => !!v && typeof v === 'object' && !Array.isArray(v) && 'class_type' in (v as object),
  )
}

export default function ComfyTemplateEditor({ templateId }: ComfyTemplateEditorProps) {
  const idStr = String(templateId)
  const [detail, setDetail] = useState<ComfyTemplateDetail | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [health, setHealth] = useState<ComfyHealth | null>(null)
  const [objectInfo, setObjectInfo] = useState<Record<string, unknown> | null>(null)
  const [exposedParams, setExposedParams] = useState<ExposedParamDraft[]>([])
  const [staleKeys, setStaleKeys] = useState<string[]>([])
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setDetail(null)
    setLoadError(null)
    setExposedParams([])
    setStaleKeys([])
    Promise.all([
      getComfyTemplate(templateId),
      getComfyHealth().catch(() => null),
      getObjectInfo().catch(() => null),
    ])
      .then(([tpl, h, info]) => {
        if (cancelled) return
        setDetail(tpl)
        setExposedParams(tpl.exposed_params ?? [])
        setHealth(h)
        setObjectInfo(info)
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [templateId])

  const workflow = (detail?.workflow_json ?? {}) as ComfyWorkflow
  const idSet = useMemo(() => new Set(Object.keys(workflow)), [workflow])

  const layout = useMemo(() => layoutComfyGraph(workflow), [workflow])
  const usedCountByNode = useMemo(() => {
    const m = new Map<string, number>()
    for (const p of exposedParams) m.set(p.comfy_node_id, (m.get(p.comfy_node_id) ?? 0) + 1)
    return m
  }, [exposedParams])
  const graphNodes = useMemo(
    () => layout.nodes.map((n) => ({ ...n, usedCount: usedCountByNode.get(n.id) ?? 0 })),
    [layout.nodes, usedCountByNode],
  )

  const rawRowsFor = (nodeId: string): RawInputRow[] => {
    const node = workflow[nodeId]
    if (!node) return []
    return Object.entries(node.inputs ?? {})
      .filter(([, v]) => !isComfyNodeRef(v, idSet))
      .map(([inputName, rawValue]) => ({ inputName, rawValue }))
  }

  const activeNode = activeNodeId ? workflow[activeNodeId] : null
  const activeRows = activeNodeId ? rawRowsFor(activeNodeId) : []
  const exposedByInput = useMemo(() => {
    const m = new Map<string, ExposedParamDraft>()
    if (!activeNodeId) return m
    for (const p of exposedParams) if (p.comfy_node_id === activeNodeId) m.set(p.comfy_input, p)
    return m
  }, [exposedParams, activeNodeId])

  const toggleExpose = (nodeId: string, inputName: string, rawValue: unknown) => {
    setExposedParams((prev) => {
      const existing = prev.find((p) => p.comfy_node_id === nodeId && p.comfy_input === inputName)
      if (existing) return prev.filter((p) => p !== existing)
      const classType = workflow[nodeId]?.class_type ?? ''
      const meta = deriveFieldMeta(classType, inputName, rawValue, objectInfo)
      const keysInUse = new Set(prev.map((p) => p.key))
      const key = keysInUse.has(inputName) ? `${inputName}_${nodeId}` : inputName
      const draft: ExposedParamDraft = {
        key,
        label: inputName,
        type: meta.type,
        comfy_node_id: nodeId,
        comfy_input: inputName,
        default: rawValue,
        required: true,
        ...(meta.min !== undefined ? { min: meta.min } : {}),
        ...(meta.max !== undefined ? { max: meta.max } : {}),
        ...(meta.step !== undefined ? { step: meta.step } : {}),
        ...(meta.options ? { options: meta.options } : {}),
      }
      return [...prev, draft]
    })
    setSaved(false)
  }

  const patchExposed = (nodeId: string, inputName: string, patch: Partial<ExposedParamDraft>) => {
    setExposedParams((prev) =>
      prev.map((p) => (p.comfy_node_id === nodeId && p.comfy_input === inputName ? { ...p, ...patch } : p)),
    )
    setSaved(false)
  }

  const removeExposed = (key: string) => {
    setExposedParams((prev) => prev.filter((p) => p.key !== key))
    setSaved(false)
  }

  const save = async () => {
    setSaving(true)
    setSaved(false)
    setSaveError(null)
    try {
      await putMapping(idStr, exposedParams)
      setSaved(true)
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleReuploadFile = (file: File) => {
    setUploadError(null)
    const reader = new FileReader()
    reader.onerror = () => setUploadError('读取文件失败')
    reader.onload = async () => {
      let parsed: unknown
      try {
        parsed = JSON.parse(String(reader.result))
      } catch {
        setUploadError('不是合法 JSON')
        return
      }
      if (!isApiFormatWorkflow(parsed)) {
        setUploadError('不是 ComfyUI API 格式导出，请在 ComfyUI 用 Export (API) 导出')
        return
      }
      try {
        const res = await reuploadComfyTemplate(idStr, parsed)
        setDetail((d) => (d ? { ...d, workflow_json: parsed } : d))
        setStaleKeys(res.stale_keys)
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : String(e))
      }
    }
    reader.readAsText(file)
  }

  if (loadError) {
    return <div style={{ fontSize: 12, color: 'var(--error, #ef4444)', padding: 14 }}>{loadError}</div>
  }
  if (!detail) {
    return <div style={{ fontSize: 12, color: 'var(--muted)', padding: 14 }}>加载中…</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <SidecarStatusLine health={health} />
      <ReuploadZone onFile={handleReuploadFile} error={uploadError} />

      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '10px 14px', background: 'var(--bg-accent)',
          border: '1px solid var(--border)', borderRadius: 8,
        }}
      >
        <div style={{ flex: 1, fontSize: 11, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertTriangle size={12} style={{ color: 'var(--warn, #f59e0b)', flexShrink: 0 }} />
          点节点卡配置要暴露的字段 · 改动会更新对外 API schema
        </div>
        {saved && !saving && <span style={{ fontSize: 11, color: 'var(--ok, #34c759)' }}>已保存</span>}
        {saveError && <span style={{ fontSize: 11, color: 'var(--error, #ef4444)' }}>{saveError}</span>}
        <button type="button" onClick={save} disabled={saving} style={saveBtnStyle(saving)}>
          {saving ? '保存中…' : '保存配置'}
        </button>
      </div>

      <div style={{ height: '58vh', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', position: 'relative' }}>
        <ComfyTemplateGraph
          nodes={graphNodes}
          edges={layout.edges}
          activeNodeId={activeNodeId}
          onNodeClick={setActiveNodeId}
        />
        {activeNodeId && activeNode && (
          <NodeConfigPopover
            nodeId={activeNodeId}
            classType={activeNode.class_type}
            rows={activeRows}
            exposedByInput={exposedByInput}
            objectInfo={objectInfo}
            onToggle={(inputName, rawValue) => toggleExpose(activeNodeId, inputName, rawValue)}
            onPatch={(inputName, patch) => patchExposed(activeNodeId, inputName, patch)}
            onClose={() => setActiveNodeId(null)}
          />
        )}
      </div>

      <ExposedSummaryTable params={exposedParams} staleKeys={staleKeys} onRemove={removeExposed} />
    </div>
  )
}

// ---------- sidecar status ----------

function SidecarStatusLine({ health }: { health: ComfyHealth | null }) {
  const lineStyle: React.CSSProperties = {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 14px', background: 'var(--bg-accent)',
    border: '1px solid var(--border)', borderRadius: 8,
    fontSize: 12, color: 'var(--text)',
  }
  if (!health) {
    return (
      <div style={lineStyle}>
        <Dot color="var(--muted)" />
        检测 sidecar 状态…
      </div>
    )
  }
  if (!health.online) {
    return (
      <div style={lineStyle}>
        <Dot color="var(--error, #ef4444)" />
        sidecar 离线 — 配置仍可手工编辑,但 object_info 类型/min/max 预填不可用
      </div>
    )
  }
  return (
    <div style={lineStyle}>
      <Dot color="var(--ok, #34c759)" />
      sidecar 在线 · v{health.version || '?'} · 队列 {health.queue_depth}
    </div>
  )
}

function Dot({ color }: { color: string }) {
  return <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
}

// ---------- reupload zone ----------

function ReuploadZone({ onFile, error }: { onFile: (f: File) => void; error: string | null }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <label style={dropZoneStyle}>
        <Upload size={14} style={{ color: 'var(--muted)', flexShrink: 0 }} />
        <span style={{ fontSize: 12, color: 'var(--text)' }}>
          重新上传工作流 JSON(替换节点结构;指向旧节点的字段会标红,需确认/移除)
        </span>
        <input
          data-testid="comfy-reupload-input"
          type="file"
          accept=".json,application/json"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
            e.target.value = ''
          }}
        />
      </label>
      {error && <div style={{ fontSize: 11, color: 'var(--error, #ef4444)' }}>{error}</div>}
    </div>
  )
}

// ---------- node config popover ----------

function ValueBadge({ value }: { value: unknown }) {
  if (typeof value === 'string') {
    const v = value.length > 50 ? value.slice(0, 50) + '…' : value
    return <span style={{ color: 'var(--text)', fontWeight: 700 }}>"{v}"</span>
  }
  if (typeof value === 'number') {
    return <span style={{ color: '#0369a1', fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
  }
  if (typeof value === 'boolean') {
    return <span style={{ color: value ? '#15803d' : '#b45309', fontWeight: 800 }}>{value ? '✓ true' : '✗ false'}</span>
  }
  if (value == null) return <span style={{ color: 'var(--muted)' }}>—</span>
  return <span style={{ color: 'var(--muted)' }}>{String(value)}</span>
}

function NumField({
  label, value, disabled, onChange,
}: {
  label: string
  value: number | undefined
  disabled: boolean
  onChange: (v: number | undefined) => void
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 10, color: 'var(--muted)', flex: 1 }}>
      {label}
      <input
        type="number"
        value={value ?? ''}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
        style={{ ...smallInput, opacity: disabled ? 0.5 : 1 }}
      />
    </label>
  )
}

function NodeConfigPopover({
  nodeId, classType, rows, exposedByInput, objectInfo, onToggle, onPatch, onClose,
}: {
  nodeId: string
  classType: string
  rows: RawInputRow[]
  exposedByInput: Map<string, ExposedParamDraft>
  objectInfo: Record<string, unknown> | null
  onToggle: (inputName: string, rawValue: unknown) => void
  onPatch: (inputName: string, patch: Partial<ExposedParamDraft>) => void
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <>
      <div onClick={onClose} style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.28)', zIndex: 20 }} />
      <div
        role="dialog"
        style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
          width: 420, maxWidth: 'calc(100% - 32px)', maxHeight: 'calc(100% - 48px)',
          display: 'flex', flexDirection: 'column', zIndex: 21,
          background: 'var(--card)', border: '1px solid var(--border)',
          borderRadius: 14, boxShadow: 'var(--shadow-lg, 0 12px 40px rgba(0,0,0,0.25))',
          overflow: 'hidden',
        }}
      >
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px',
          borderBottom: '1px solid var(--border)', background: 'var(--card-hl)',
        }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {classType}
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)' }}>#{nodeId}</div>
          </div>
          <button
            type="button" title="关闭" onClick={onClose}
            style={{ display: 'inline-flex', padding: 4, background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer', borderRadius: 6 }}
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ overflow: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {rows.length === 0 && (
            <div style={{ color: 'var(--muted)', fontSize: 12, padding: '12px 4px' }}>该节点无可配置的原始字段</div>
          )}
          {rows.map((r) => {
            const cur = exposedByInput.get(r.inputName)
            const active = !!cur
            const meta = deriveFieldMeta(classType, r.inputName, r.rawValue, objectInfo)
            const type = cur?.type ?? meta.type
            const numeric = NUMERIC_TYPES.has(type)
            const seedLike = isSeedLike(r.inputName, type)
            return (
              <div
                key={r.inputName}
                data-input-row
                style={{
                  border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 10, padding: 10,
                  background: active ? 'var(--accent-subtle, rgba(99,102,241,0.06))' : 'transparent',
                  display: 'flex', flexDirection: 'column', gap: 8,
                }}
              >
                <label style={{ display: 'flex', alignItems: 'flex-start', gap: 9, cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    aria-label={`暴露 ${r.inputName}`}
                    checked={active}
                    onChange={() => onToggle(r.inputName, r.rawValue)}
                    style={{ marginTop: 2, cursor: 'pointer' }}
                  />
                  <span style={{ minWidth: 0, flex: 1 }}>
                    <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>{r.inputName}</span>
                    <span style={{ display: 'block', fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
                      原值: <ValueBadge value={r.rawValue} />
                    </span>
                  </span>
                </label>

                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    style={{ ...smallInput, flex: 1, minWidth: 0, opacity: active ? 1 : 0.5 }}
                    placeholder="显示名"
                    value={(cur?.label ?? r.inputName) || ''}
                    disabled={!active}
                    onChange={(e) => onPatch(r.inputName, { label: e.target.value })}
                  />
                  <select
                    style={{ ...smallInput, width: 84, flexShrink: 0, opacity: active ? 1 : 0.5 }}
                    value={type}
                    disabled={!active}
                    onChange={(e) => onPatch(r.inputName, { type: e.target.value })}
                  >
                    {TYPE_OPTIONS.map((o) => (
                      <option key={o.v} value={o.v}>{o.label}</option>
                    ))}
                  </select>
                </div>

                {numeric && (
                  <div style={{ display: 'flex', gap: 8 }}>
                    <NumField
                      label="min" value={cur?.min ?? meta.min} disabled={!active}
                      onChange={(v) => onPatch(r.inputName, { min: v })}
                    />
                    <NumField
                      label="max" value={cur?.max ?? meta.max} disabled={!active}
                      onChange={(v) => onPatch(r.inputName, { max: v })}
                    />
                    <NumField
                      label="step" value={cur?.step ?? meta.step} disabled={!active}
                      onChange={(v) => onPatch(r.inputName, { step: v })}
                    />
                  </div>
                )}

                {seedLike && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text)', opacity: active ? 1 : 0.5 }}>
                    <input
                      type="checkbox"
                      aria-label={`随机 ${r.inputName}`}
                      checked={!!cur?.random}
                      disabled={!active}
                      onChange={(e) => onPatch(r.inputName, { random: e.target.checked })}
                    />
                    每次调用随机取值(不由调用方传入)
                  </label>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}

// ---------- summary table ----------

function ExposedSummaryTable({
  params, staleKeys, onRemove,
}: {
  params: ExposedParamDraft[]
  staleKeys: string[]
  onRemove: (key: string) => void
}) {
  const staleSet = useMemo(() => new Set(staleKeys), [staleKeys])
  if (params.length === 0) {
    return (
      <div style={{
        fontSize: 12, color: 'var(--muted)', padding: '10px 14px',
        border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-accent)',
      }}>
        暂无暴露字段 — 在上方节点图点节点卡配置
      </div>
    )
  }
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'auto', background: 'var(--bg-accent)' }}>
      <table data-testid="exposed-summary-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ color: 'var(--muted)', fontSize: 11, textAlign: 'left' }}>
            <th style={th}>key</th>
            <th style={th}>显示名</th>
            <th style={th}>type</th>
            <th style={th}>节点</th>
            <th style={th}>输入</th>
            <th style={th} />
          </tr>
        </thead>
        <tbody>
          {params.map((p) => {
            const stale = staleSet.has(p.key)
            return (
              <tr
                key={p.key}
                data-testid={`exposed-row-${p.key}`}
                data-stale={stale ? 'true' : 'false'}
                style={{
                  borderTop: '1px solid var(--border)',
                  background: stale ? 'rgba(239,68,68,0.1)' : undefined,
                }}
              >
                <td style={td}><code style={{ fontFamily: 'var(--mono, monospace)' }}>{p.key}</code></td>
                <td style={{ ...td, color: 'var(--muted)' }}>{p.label || '—'}</td>
                <td style={td}>{p.type}</td>
                <td style={td}><code style={{ fontFamily: 'var(--mono, monospace)', fontSize: 11 }}>{p.comfy_node_id}</code></td>
                <td style={td}><code style={{ fontFamily: 'var(--mono, monospace)', fontSize: 11 }}>{p.comfy_input}</code></td>
                <td style={td}>
                  <button
                    type="button" onClick={() => onRemove(p.key)} title="移除暴露"
                    style={{ background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer', padding: 2 }}
                  >
                    <X size={12} />
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {staleKeys.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--error, #ef4444)', padding: '6px 12px', borderTop: '1px solid var(--border)' }}>
          重新上传后 {staleKeys.length} 个字段指向的节点/输入已不存在(标红行)— 修正或移除后再保存
        </div>
      )}
    </div>
  )
}

// ---------- shared bits ----------

const smallInput: React.CSSProperties = {
  fontSize: 12, padding: '6px 8px',
  background: 'var(--bg)', color: 'var(--text)',
  border: '1px solid var(--border)', borderRadius: 6,
}

const dropZoneStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8, width: '100%',
  background: 'var(--bg-accent)', color: 'var(--text)',
  border: '1px dashed var(--border)', borderRadius: 8,
  padding: '10px 12px', fontSize: 12, cursor: 'pointer',
}

const th = { padding: '6px 10px', fontWeight: 500 } as const
const td = { padding: '6px 10px', color: 'var(--text)' } as const

function saveBtnStyle(saving: boolean): React.CSSProperties {
  return {
    fontSize: 12, padding: '6px 14px', background: 'var(--accent)', color: '#fff',
    border: 'none', borderRadius: 4, cursor: saving ? 'not-allowed' : 'pointer',
    opacity: saving ? 0.6 : 1,
  }
}
