import { useMemo, useState, type ReactNode } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Code2,
  KeyRound,
  LayoutGrid,
  Pause,
  Pencil,
  Play,
  Plus,
  SlidersHorizontal,
  Trash2,
  Unlink,
} from 'lucide-react'
import {
  endpointFor,
  NAME_RE,
  useDeleteService,
  usePatchService,
  useService,
  type ExposedParam,
  type ServiceDetail as ServiceDetailT,
  type ServiceModelRef,
  type ServiceStatus,
} from '../api/services'
import WorkflowAppEditor, { type AppEditorValue } from '../components/workflow/WorkflowAppEditor'
import type { EditorNodeLike } from '../components/workflow/appEditorSchema'
import { useServiceModelStatus, MODEL_STATE_VIS, MODEL_ROLE_LABEL } from '../api/serviceModels'
import {
  useApiKeys,
  useAddGrant,
  useRemoveGrant,
  useServiceGrants,
  useToggleGrant,
} from '../api/keys'
import CreateApiKeyDialog from '../components/api-keys/CreateApiKeyDialog'
import SchemaDrivenForm from '../components/playground/SchemaDrivenForm'
import SchemaDrivenOutput from '../components/playground/SchemaDrivenOutput'
import { useTasksByWorkflow, type ExecutionTask } from '../api/tasks'
import { apiFetch } from '../api/client'
import { confirmDialog } from '../stores/confirm'

export interface ServiceDetailPageProps {
  serviceId: string
  onBack?: () => void
}

// Playground(运行)与应用编辑(配置暴露字段)合并成一个 tab,内部「运行/编辑」
// 切换(对齐 Infinite-Canvas 单模块 + 工作流/测试画布切换)。LLM/TTS 无工作流图 →
// 只有运行视图。
const TABS = [
  { id: 'overview', label: '总览', icon: LayoutGrid },
  { id: 'playground', label: 'Playground', icon: Play },
  { id: 'docs', label: 'API 文档', icon: Code2 },
  { id: 'auth', label: 'Key 授权', icon: KeyRound },
  { id: 'usage', label: '用量', icon: Activity },
] as const

type TabId = (typeof TABS)[number]['id']

export default function ServiceDetailPage({ serviceId, onBack }: ServiceDetailPageProps) {
  const { data: svc, isLoading, error } = useService(serviceId)
  const location = useLocation()
  // 从任务面板「重跑(相同参数)」跳来时带的历史入参 → 预填 Playground 表单。
  const rerunInputs = (location.state as { rerunInputs?: Record<string, unknown> } | null)?.rerunInputs
  const [tab, setTab] = useState<TabId>('playground')

  if (isLoading) {
    return <FullPageMessage message="加载中…" />
  }
  if (error) {
    return <FullPageMessage message={(error as Error).message} variant="error" />
  }
  if (!svc) return null

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'auto',
        background: 'var(--bg)',
      }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20 }}>
        <Breadcrumb name={svc.name} onBack={onBack} />
        <Header svc={svc} />
        <Tabs current={tab} onChange={setTab} />

        {tab === 'overview' && <OverviewTab svc={svc} />}
        {tab === 'playground' && <AppTab svc={svc} initialInputs={rerunInputs} />}
        {tab === 'docs' && <DocsTab svc={svc} />}
        {tab === 'auth' && <AuthTab svc={svc} />}
        {tab === 'usage' && <UsageTab svc={svc} />}
      </div>
    </div>
  )
}

function Breadcrumb({ name, onBack }: { name: string; onBack?: () => void }) {
  return (
    <div style={{
      fontSize: 12, color: 'var(--muted)', marginBottom: 8,
      display: 'flex', alignItems: 'center',
    }}>
      <button
        type="button"
        onClick={onBack}
        style={{
          background: 'transparent',
          border: '1px solid transparent',
          color: 'var(--muted)',
          cursor: 'pointer',
          padding: '2px 8px',
          marginLeft: -8,
          borderRadius: 4,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          lineHeight: 1,
          transition: 'color 0.12s, border-color 0.12s, background 0.12s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = 'var(--accent)'
          e.currentTarget.style.borderColor = 'var(--border)'
          e.currentTarget.style.background = 'var(--accent-subtle, rgba(99,102,241,0.08))'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = 'var(--muted)'
          e.currentTarget.style.borderColor = 'transparent'
          e.currentTarget.style.background = 'transparent'
        }}
      >
        <ArrowLeft size={12} />
        <span>服务</span>
      </button>
      <span style={{ margin: '0 6px' }}>/</span>
      <span style={{ color: 'var(--text)' }}>{name}</span>
    </div>
  )
}

function Header({ svc }: { svc: ServiceDetailT }) {
  const patch = usePatchService()
  const del = useDeleteService()

  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(svc.name)
  const [err, setErr] = useState<string | null>(null)

  const setStatus = (s: ServiceStatus) =>
    patch.mutate({ serviceId: svc.id, status: s })

  const startEdit = () => {
    setDraft(svc.name)
    setErr(null)
    setEditing(true)
  }

  const saveRename = async () => {
    const next = draft.trim()
    if (next === svc.name) {
      setEditing(false)
      return
    }
    if (!NAME_RE.test(next)) {
      setErr('格式:小写字母开头,仅小写字母/数字/连字符,长度 2-63')
      return
    }
    const ok = await confirmDialog({
      message:
        `把服务名 "${svc.name}" 改为 "${next}"?\n\n` +
        `⚠️ 这是对外调用的路由键(model / 端点路径)。改名后,仍用旧名 ` +
        `"${svc.name}" 调用的客户端会收到 404,需同步更新调用方代码。\n` +
        `(已授权的 API Key 不受影响 — grant 按服务 ID 绑定。)`,
      danger: true,
      confirmText: '改名',
    })
    if (!ok) return
    patch.mutate(
      { serviceId: svc.id, name: next },
      {
        onSuccess: () => setEditing(false),
        onError: (e) =>
          setErr(e instanceof Error ? e.message : '改名失败(名称可能已被占用)'),
      },
    )
  }

  return (
    <div
      style={{
        display: 'flex',
        gap: 16,
        marginBottom: 14,
        paddingBottom: 14,
        borderBottom: '1px solid var(--border)',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {editing ? (
            <>
              <input
                autoFocus
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value)
                  setErr(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveRename()
                  if (e.key === 'Escape') setEditing(false)
                }}
                style={{
                  fontSize: 20,
                  fontFamily: 'var(--mono, monospace)',
                  background: 'var(--bg)',
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '4px 8px',
                  minWidth: 220,
                }}
              />
              <SmallBtn onClick={saveRename} icon={Pencil}>
                保存
              </SmallBtn>
              <SmallBtn onClick={() => setEditing(false)}>取消</SmallBtn>
            </>
          ) : (
            <>
              <h1 style={{ fontSize: 22, color: 'var(--text)' }}>{svc.name}</h1>
              <button
                onClick={startEdit}
                title="改名(对外路由键)"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--muted)',
                  cursor: 'pointer',
                  padding: 2,
                }}
              >
                <Pencil size={15} />
              </button>
            </>
          )}
          <StatusBadge status={svc.status} />
          <SourceBadge svc={svc} />
        </div>
        {err && (
          <div style={{ color: 'var(--danger, #ef4444)', fontSize: 11, marginTop: 6 }}>{err}</div>
        )}
        <div
          style={{
            display: 'flex',
            gap: 18,
            fontSize: 11,
            color: 'var(--muted)',
            marginTop: 8,
          }}
        >
          <span>
            endpoint: <b style={{ color: 'var(--text)' }}>{endpointFor(svc)}</b>
          </span>
          {svc.workflow_id && (
            <span>
              源 Workflow: <b style={{ color: 'var(--text)' }}>{svc.workflow_id}</b>
            </span>
          )}
          {svc.snapshot_hash && (
            <span style={{ fontFamily: 'var(--mono, monospace)' }}>
              snapshot: {svc.snapshot_hash.slice(7, 19)}…
            </span>
          )}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
        {svc.status === 'active' ? (
          <SmallBtn onClick={() => setStatus('paused')} icon={Pause}>
            暂停
          </SmallBtn>
        ) : (
          <SmallBtn onClick={() => setStatus('active')} icon={Play}>
            启用
          </SmallBtn>
        )}
        <SmallBtn
          onClick={async () => {
            if (await confirmDialog({ message: `下线服务 ${svc.name}?\n此操作不可逆。`, danger: true, confirmText: '下线' })) {
              del.mutate(svc.id, { onSuccess: () => history.back() })
            }
          }}
          icon={Trash2}
          danger
        >
          下线
        </SmallBtn>
      </div>
    </div>
  )
}

function Tabs({
  current,
  onChange,
}: {
  current: TabId
  onChange: (t: TabId) => void
}) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 0,
        borderBottom: '1px solid var(--border)',
        marginBottom: 16,
      }}
    >
      {TABS.map(({ id, label, icon: Icon }) => {
        const active = current === id
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(id)}
            style={{
              padding: '10px 18px',
              fontSize: 13,
              color: active ? 'var(--text)' : 'var(--muted)',
              background: 'transparent',
              border: 'none',
              borderBottom: '2px solid',
              borderBottomColor: active ? 'var(--accent)' : 'transparent',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontWeight: active ? 500 : 400,
            }}
          >
            <Icon size={14} />
            {label}
          </button>
        )
      })}
    </div>
  )
}

function OverviewTab({ svc }: { svc: ServiceDetailT }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Panel title="基本信息">
          <KV k="ID" v={svc.id} />
          <KV k="类型" v={svc.type} />
          <KV k="状态" v={svc.status} />
          <KV k="创建时间" v={new Date(svc.created_at).toLocaleString()} />
          <KV k="更新时间" v={new Date(svc.updated_at).toLocaleString()} />
        </Panel>
        <Panel title="发布快照">
          <KV k="hash" v={svc.snapshot_hash ?? '—'} mono />
          <KV k="schema 版本" v={String(svc.snapshot_schema_version)} />
          <KV k="服务版本" v={`v${svc.version}`} />
          <KV k="入参数量" v={String(svc.exposed_inputs.length)} />
          <KV k="出参数量" v={String(svc.exposed_outputs.length)} />
        </Panel>
      </div>
      <ModelsPanel models={svc.models} />
    </div>
  )
}

function ModelsPanel({ models }: { models: ServiceModelRef[] }) {
  const { refs, total, loaded, loading, failed } = useServiceModelStatus(models)
  return (
    <Panel title={`模型 — 已加载 ${loaded}/${total}${loading ? ` · 加载中 ${loading}` : ''}${failed ? ` · 失败 ${failed}` : ''}`}>
      {total === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 14px' }}>
          — 该服务的工作流未引用可追踪的模型/组件 —
        </div>
      ) : (
        <div>
          {refs.map((r, i) => {
            const vis = MODEL_STATE_VIS[r.state]
            return (
              <div
                key={`${r.kind}:${r.file ?? r.engine_key ?? i}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 14px',
                  borderTop: i === 0 ? 'none' : '1px solid var(--border)',
                  fontSize: 12,
                }}
              >
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: vis.color, flexShrink: 0 }} />
                <span
                  style={{
                    fontSize: 10,
                    color: 'var(--muted)',
                    width: 78,
                    flexShrink: 0,
                    textTransform: 'uppercase',
                    letterSpacing: 0.4,
                  }}
                >
                  {(r.role && MODEL_ROLE_LABEL[r.role]) ?? r.kind}
                </span>
                <span
                  style={{
                    flex: 1,
                    minWidth: 0,
                    color: 'var(--text)',
                    fontFamily: 'var(--mono, monospace)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={r.file ?? r.engine_key ?? r.label}
                >
                  {r.label}
                </span>
                <span style={{ fontSize: 11, color: vis.color, flexShrink: 0 }}>{vis.label}</span>
              </div>
            )
          })}
        </div>
      )}
    </Panel>
  )
}

// 合并后的「Playground」tab:运行(干净表单+输出)与编辑(节点图+暴露字段配置)
// 二合一,顶部分段切换。仅工作流类服务(有节点快照)才出现「编辑」;LLM/TTS 只有运行。
function AppTab({ svc, initialInputs }: { svc: ServiceDetailT; initialInputs?: Record<string, unknown> }) {
  const canEdit = useMemo(
    () => snapshotToNodes(svc.workflow_snapshot ?? {}).length > 0,
    [svc.workflow_snapshot],
  )
  const [mode, setMode] = useState<'run' | 'edit'>('run')
  const m = canEdit ? mode : 'run'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {canEdit && (
        <div style={{ display: 'inline-flex', alignSelf: 'flex-start', gap: 2, padding: 3, background: 'var(--bg-accent)', border: '1px solid var(--border)', borderRadius: 8 }}>
          <SegBtn active={m === 'run'} onClick={() => setMode('run')} icon={<Play size={13} />} label="运行" />
          <SegBtn active={m === 'edit'} onClick={() => setMode('edit')} icon={<SlidersHorizontal size={13} />} label="编辑暴露字段" />
        </div>
      )}
      {m === 'edit' ? <AppEditorTab svc={svc} /> : <PlaygroundTab svc={svc} initialInputs={initialInputs} />}
    </div>
  )
}

function SegBtn({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: ReactNode; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 6,
        fontSize: 12, fontWeight: 600, padding: '6px 12px', borderRadius: 6,
        border: 'none', cursor: 'pointer',
        background: active ? 'var(--card)' : 'transparent',
        color: active ? 'var(--text)' : 'var(--muted)',
        boxShadow: active ? 'var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.1))' : 'none',
      }}
    >
      {icon}{label}
    </button>
  )
}

function PlaygroundTab({ svc, initialInputs }: { svc: ServiceDetailT; initialInputs?: Record<string, unknown> }) {
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle')
  // ASR(语音识别):音频进文本出,走 multipart /v1/audio/transcriptions —— 没有 exposed_inputs
  // 表单可填,改成传音频文件(2026-06-21:用户反馈 ASR 服务没法在 Playground 测)。
  const isAsr = svc.category === 'asr'
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)

  // 拖拽/选择共用:只收音频文件(部分音频 type 为空,放行靠扩展名兜底)。
  const acceptAudio = (f: File | null | undefined) => {
    if (!f) return
    const okType = f.type.startsWith('audio/')
    const okExt = /\.(wav|mp3|m4a|flac|ogg|opus|aac|webm|mp4)$/i.test(f.name)
    if (!okType && !okExt) {
      setError(`不是音频文件:${f.name}`)
      return
    }
    setError(null)
    setAudioFile(f)
  }

  const submit = async (values: Record<string, unknown>) => {
    setRunning(true)
    setError(null)
    setResult(null)
    setLatencyMs(null)
    setStatus('running')
    const start = performance.now()
    try {
      let data: Record<string, unknown>
      if (isAsr) {
        if (!audioFile) throw new Error('请先选择音频文件')
        // multipart 不能走 apiFetch(它强制 JSON Content-Type);raw fetch + same-origin cookie。
        const fd = new FormData()
        fd.append('file', audioFile)
        fd.append('model', svc.name)
        const resp = await fetch('/v1/audio/transcriptions', {
          method: 'POST', body: fd, credentials: 'same-origin',
        })
        if (!resp.ok) {
          const t = await resp.text().catch(() => '')
          throw new Error(`HTTP ${resp.status}: ${t.slice(0, 200)}`)
        }
        data = await resp.json()
      } else {
        const url =
          svc.category === 'llm'
            ? '/v1/chat/completions'
            : `/v1/apps/${encodeURIComponent(svc.name)}/run`
        const body = svc.category === 'llm' ? buildLlmBody(svc, values) : values
        data = await apiFetch<Record<string, unknown>>(url, {
          method: 'POST',
          body: JSON.stringify(body),
        })
      }
      setResult(data)
      setStatus('completed')
      setLatencyMs(Math.round(performance.now() - start))
    } catch (e) {
      const msg = (e as Error).message ?? String(e)
      setError(msg)
      setStatus('failed')
      setLatencyMs(Math.round(performance.now() - start))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Stage 1: Input form */}
      <section style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <SectionHeader title={isAsr ? '音频输入' : '入参'} />
        {isAsr ? (
          <div style={{ padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label
              onDragOver={(e) => { e.preventDefault(); if (!dragging) setDragging(true) }}
              onDragLeave={(e) => { e.preventDefault(); setDragging(false) }}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                acceptAudio(e.dataTransfer.files?.[0])
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '12px 14px',
                borderRadius: 6,
                border: `1px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
                background: dragging ? 'var(--accent-subtle, rgba(99,102,241,0.1))' : 'var(--bg)',
                color: 'var(--muted)', fontSize: 13, cursor: 'pointer',
                transition: 'border-color 120ms, background 120ms',
              }}
            >
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => acceptAudio(e.target.files?.[0])}
                style={{ display: 'none' }}
              />
              {dragging
                ? <span style={{ color: 'var(--accent)' }}>松开以上传音频</span>
                : audioFile
                  ? <span style={{ color: 'var(--text)' }}>{audioFile.name} · {(audioFile.size / 1024).toFixed(0)} KB</span>
                  : <span>拖拽音频到此,或点击选择(wav / mp3 / m4a…)转写为文本</span>}
            </label>
            <button
              type="button"
              disabled={running || !audioFile}
              onClick={() => submit({})}
              style={{
                alignSelf: 'flex-start', padding: '7px 16px', borderRadius: 6,
                background: running || !audioFile ? 'var(--muted)' : 'var(--accent)',
                color: '#fff', border: 'none', fontSize: 13, fontWeight: 500,
                cursor: running || !audioFile ? 'not-allowed' : 'pointer',
              }}
            >
              {running ? '转写中…' : '▶ 运行'}
            </button>
          </div>
        ) : (
          <SchemaDrivenForm
            inputs={svc.exposed_inputs}
            initialValues={initialInputs}
            submitting={running}
            onSubmit={submit}
          />
        )}
      </section>

      {/* Stage 2: Output (takes focus when there's content) */}
      {(status !== 'idle' || result || error) && (
        <section style={{
          background: 'var(--bg-accent)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}>
          <SectionHeader title="输出" />
          <div style={{ padding: '16px 18px' }}>
            {status === 'running' && !result && !error && (
              <div style={{
                color: 'var(--muted)', fontSize: 12, padding: '40px 16px',
                textAlign: 'center',
              }}>
                运行中…
              </div>
            )}
            {isAsr ? (
              error ? (
                <div style={{ color: 'var(--danger, #ef4444)', fontSize: 13 }}>{error}</div>
              ) : result && typeof result.text === 'string' ? (
                <div style={{ fontSize: 14, color: 'var(--text)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {result.text as string}
                </div>
              ) : null
            ) : (
              <SchemaDrivenOutput
                outputs={svc.exposed_outputs}
                result={result}
                error={error}
              />
            )}
          </div>
          {/* Stage 3: Status footer */}
          {status !== 'idle' && (
            <StatusFooter status={status} latencyMs={latencyMs} result={result} />
          )}
        </section>
      )}
    </div>
  )
}

// 服务页「应用编辑」tab(spec 2026-06-09 PR-4)。对应用户图3「加个 tab 来修改」+
// 图4 Infinite-Canvas 测试画布。在已发布服务上改逐 widget 暴露 schema,保存走
// PATCH(PR-1);左侧表单可真跑(runnable,复用 /v1/apps/{name}/run)。
function snapshotToNodes(snapshot: Record<string, unknown>): EditorNodeLike[] {
  const raw = snapshot?.nodes
  if (Array.isArray(raw)) {
    return raw.map((n) => {
      const node = n as Record<string, unknown>
      return {
        id: String(node.id),
        type: String(node.type || node.class_type || ''),
        data: (node.data || node.inputs || {}) as Record<string, unknown>,
        position: node.position as { x: number; y: number } | undefined,
      }
    })
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw as Record<string, Record<string, unknown>>).map(([id, n]) => ({
      id,
      type: String(n.class_type || n.type || ''),
      data: (n.inputs || n.data || {}) as Record<string, unknown>,
    }))
  }
  return []
}

function snapshotEdges(snapshot: Record<string, unknown>): Array<{ source: string; target: string }> {
  const raw = snapshot?.edges
  if (!Array.isArray(raw)) return []
  return raw
    .map((e) => e as Record<string, unknown>)
    .filter((e) => e.source && e.target)
    .map((e) => ({ source: String(e.source), target: String(e.target) }))
}

function AppEditorTab({ svc }: { svc: ServiceDetailT }) {
  const patch = usePatchService()
  const nodes = useMemo(() => snapshotToNodes(svc.workflow_snapshot ?? {}), [svc.workflow_snapshot])
  const edges = useMemo(() => snapshotEdges(svc.workflow_snapshot ?? {}), [svc.workflow_snapshot])
  // 草稿:从服务当前 exposed 播种;改完点保存才落库。
  const [draft, setDraft] = useState<AppEditorValue>({
    inputs: svc.exposed_inputs ?? [],
    outputs: svc.exposed_outputs ?? [],
  })
  const [saved, setSaved] = useState(false)

  // 运行(可真跑):复用外部 app 端点。
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [runErr, setRunErr] = useState<string | null>(null)

  const run = async (values: Record<string, unknown>) => {
    setRunning(true)
    setResult(null)
    setRunErr(null)
    try {
      const data = await apiFetch<Record<string, unknown>>(
        `/v1/apps/${encodeURIComponent(svc.name)}/run`,
        { method: 'POST', body: JSON.stringify(values) },
      )
      setResult(data)
    } catch (e) {
      setRunErr((e as Error).message ?? String(e))
    } finally {
      setRunning(false)
    }
  }

  const save = () => {
    setSaved(false)
    patch.mutate(
      { serviceId: svc.id, exposed_inputs: draft.inputs, exposed_outputs: draft.outputs },
      { onSuccess: () => setSaved(true) },
    )
  }

  if (nodes.length === 0) {
    return (
      <Panel title="应用编辑">
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: '12px 14px' }}>
          该服务的工作流快照为空,无法编辑暴露字段。
        </div>
      </Panel>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* 工具条:保存 + 契约提示(R3) */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '10px 14px', background: 'var(--bg-accent)',
        border: '1px solid var(--border)', borderRadius: 8,
      }}>
        <div style={{ flex: 1, fontSize: 11, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertTriangle size={12} style={{ color: 'var(--warn, #f59e0b)', flexShrink: 0 }} />
          在节点上勾选要暴露的参数 · 改动会更新对外 API schema,可能影响已对接的调用方
        </div>
        {saved && !patch.isPending && (
          <span style={{ fontSize: 11, color: 'var(--ok, #34c759)' }}>已保存</span>
        )}
        {patch.error && (
          <span style={{ fontSize: 11, color: 'var(--error, #ef4444)' }}>
            {(patch.error as Error).message}
          </span>
        )}
        <button
          type="button"
          onClick={save}
          disabled={patch.isPending}
          style={{
            fontSize: 12, padding: '6px 14px', background: 'var(--accent)', color: '#fff',
            border: 'none', borderRadius: 4, cursor: patch.isPending ? 'not-allowed' : 'pointer',
            opacity: patch.isPending ? 0.6 : 1,
          }}
        >
          {patch.isPending ? '保存中…' : '保存配置'}
        </button>
      </div>

      <div style={{
        height: '68vh', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden',
      }}>
        <WorkflowAppEditor
          nodes={nodes}
          edges={edges}
          value={draft}
          onChange={(v) => { setDraft(v); setSaved(false) }}
          runnable
          running={running}
          onRun={run}
          formFooter={
            (result || runErr) ? (
              <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)' }}>
                <SchemaDrivenOutput outputs={draft.outputs} result={result} error={runErr} />
              </div>
            ) : null
          }
        />
      </div>
    </div>
  )
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div style={{
      fontSize: 11, color: 'var(--muted)',
      textTransform: 'uppercase', letterSpacing: 0.5,
      padding: '10px 18px',
      borderBottom: '1px solid var(--border)',
    }}>
      {title}
    </div>
  )
}

function StatusFooter({
  status,
  latencyMs,
  result,
}: {
  status: 'running' | 'completed' | 'failed' | 'idle'
  latencyMs: number | null
  result?: Record<string, unknown> | null
}) {
  const usage = collectLlmUsage(result)
  const dotColor =
    status === 'completed' ? 'var(--ok, #34c759)' :
    status === 'failed' ? 'var(--error, #ef4444)' :
    status === 'running' ? 'var(--accent, #f43f5e)' : 'var(--muted)'
  const label =
    status === 'running' ? '运行中' :
    status === 'completed' ? 'completed' :
    status === 'failed' ? 'failed' : ''
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '8px 18px',
      borderTop: '1px solid var(--border)',
      background: 'var(--bg)',
      fontSize: 11, color: 'var(--muted)',
      fontFamily: 'var(--mono, monospace)',
    }}>
      <style>{`@keyframes pgPulse{0%,100%{opacity:1}50%{opacity:.4}}`}</style>
      <span style={{
        width: 7, height: 7, borderRadius: '50%',
        background: dotColor, flexShrink: 0,
        boxShadow: status === 'running'
          ? `0 0 0 3px color-mix(in oklab, ${dotColor} 25%, transparent)`
          : 'none',
        animation: status === 'running' ? 'pgPulse 1.4s ease-in-out infinite' : 'none',
      }} />
      <span>{label}</span>
      {latencyMs !== null && (
        <>
          <span>·</span>
          <span>{formatLatency(latencyMs)}</span>
        </>
      )}
      {usage && (
        <>
          <span>·</span>
          <span>{usage.total} tok ({usage.prompt} in / {usage.completion} out)</span>
          {usage.tps && (
            <>
              <span>·</span>
              <span>{usage.tps.toFixed(1)} tok/s</span>
            </>
          )}
        </>
      )}
    </div>
  )
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

/**
 * LLM nodes attach `{ usage: {prompt_tokens, completion_tokens, total_tokens},
 * duration_ms }` to their bucket. Find the first node that did so and surface
 * the aggregate stats. Multi-LLM flows: pick the heaviest by completion tokens.
 */
function collectLlmUsage(result?: Record<string, unknown> | null): {
  prompt: number; completion: number; total: number; tps: number | null
} | null {
  if (!result) return null
  const root = (result.outputs && typeof result.outputs === 'object'
    ? result.outputs
    : result) as Record<string, unknown>
  let best: {
    prompt: number; completion: number; total: number; tps: number | null
  } | null = null
  for (const v of Object.values(root)) {
    if (!v || typeof v !== 'object') continue
    const bucket = v as Record<string, unknown>
    const u = bucket.usage as Record<string, unknown> | undefined
    if (!u || typeof u !== 'object') continue
    const prompt = Number(u.prompt_tokens ?? 0)
    const completion = Number(u.completion_tokens ?? 0)
    const total = Number(u.total_tokens ?? prompt + completion)
    const durMs = Number(bucket.duration_ms ?? 0)
    const tps = durMs > 0 && completion > 0 ? (completion / (durMs / 1000)) : null
    const candidate = { prompt, completion, total, tps }
    if (!best || candidate.completion > best.completion) best = candidate
  }
  return best
}

function buildLlmBody(svc: ServiceDetailT, values: Record<string, unknown>): Record<string, unknown> {
  // Heuristic: if there's a single string-like input, treat it as the user
  // message for OpenAI-compat. Multiple inputs → still single message
  // joined for now (PR-B scope: enough to round-trip; agent prompt building
  // is a separate ticket).
  const text = Object.values(values)
    .filter((v) => typeof v === 'string' && (v as string).length > 0)
    .join('\n')
  return {
    model: svc.name,
    messages: text ? [{ role: 'user', content: text }] : [],
  }
}

function DocsTab({ svc }: { svc: ServiceDetailT }) {
  const curl = buildCurl(svc)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Panel title="cURL">
        <pre
          style={{
            margin: 0,
            padding: 12,
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            fontSize: 11,
            color: 'var(--text)',
            overflow: 'auto',
            fontFamily: 'var(--mono, monospace)',
          }}
        >
          {curl}
        </pre>
      </Panel>
      <Panel title="入参 schema">
        <SchemaTable rows={svc.exposed_inputs} kind="input" />
      </Panel>
      <Panel title="出参 schema">
        <SchemaTable rows={svc.exposed_outputs} kind="output" />
      </Panel>
    </div>
  )
}

function buildCurl(svc: ServiceDetailT): string {
  const url =
    svc.category === 'llm'
      ? 'https://YOUR_HOST/v1/chat/completions'
      : `https://YOUR_HOST/v1/apps/${svc.name}/run`
  const body =
    svc.category === 'llm'
      ? `{"model": "${svc.name}", "messages": [{"role": "user", "content": "..."}]}`
      : '{}'
  return [
    `curl -X POST '${url}' \\`,
    `  -H 'Authorization: Bearer YOUR_API_KEY' \\`,
    `  -H 'Content-Type: application/json' \\`,
    `  -d '${body}'`,
  ].join('\n')
}

function SchemaTable({ rows, kind }: { rows: ExposedParam[]; kind: 'input' | 'output' }) {
  if (rows.length === 0) {
    return (
      <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 12px' }}>
        — 无 {kind === 'input' ? '入参' : '出参'} —
      </div>
    )
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
      <thead>
        <tr style={{ color: 'var(--muted)', fontSize: 11, textAlign: 'left' }}>
          <th style={th}>key</th>
          <th style={th}>type</th>
          <th style={th}>node_id</th>
          <th style={th}>label</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((p, i) => (
          <tr key={`${p.node_id}.${i}`} style={{ borderTop: '1px solid var(--border)' }}>
            <td style={td}>
              <code style={{ fontFamily: 'var(--mono, monospace)' }}>
                {p.key ?? p.api_name ?? '—'}
              </code>
            </td>
            <td style={td}>{p.type ?? 'string'}</td>
            <td style={td}>
              <code style={{ fontFamily: 'var(--mono, monospace)', fontSize: 11 }}>{p.node_id}</code>
            </td>
            <td style={{ ...td, color: 'var(--muted)' }}>{p.label ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

const th = { padding: '6px 10px', fontWeight: 500 } as const
const td = { padding: '6px 10px', color: 'var(--text)' } as const

function AuthTab({ svc }: { svc: ServiceDetailT }) {
  return <RealAuthTab serviceId={svc.id} serviceName={svc.name} />
}

function UsageTab({ svc }: { svc: ServiceDetailT }) {
  // 该服务源 workflow 的历史调用记录(service run 的 task 经 PR-A 已带 workflow_id)。
  const { data: tasks, isLoading } = useTasksByWorkflow(svc.workflow_id)
  const [openId, setOpenId] = useState<string | null>(null)

  if (!svc.workflow_id) {
    return (
      <Panel title="用量 / 历史">
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 14px' }}>
          该服务非工作流来源,暂无可归属的调用历史。
        </div>
      </Panel>
    )
  }

  const rows = tasks ?? []
  const total = rows.length
  const done = rows.filter((t) => t.status === 'completed').length
  const failed = rows.filter((t) => t.status === 'failed').length
  const durs = rows.map((t) => t.duration_ms).filter((d): d is number => d != null)
  const avgMs = durs.length ? Math.round(durs.reduce((a, b) => a + b, 0) / durs.length) : null
  const successPct = total ? Math.round((done / total) * 100) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Panel title="概览 · 最近 50 次调用">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0 }}>
          <KV k="调用数" v={String(total)} />
          <KV k="成功率" v={successPct == null ? '—' : `${successPct}%`} />
          <KV k="均耗时" v={avgMs == null ? '—' : avgMs < 1000 ? `${avgMs} ms` : `${(avgMs / 1000).toFixed(2)} s`} />
          <KV k="失败" v={String(failed)} />
        </div>
        <div style={{ fontSize: 10, color: 'var(--muted)', padding: '4px 14px 0' }}>
          按 API key 细分待用量子系统接入(ExecutionTask 暂无 api_key 归属字段)。
        </div>
      </Panel>

      <Panel title="历史调用">
        {isLoading ? (
          <div style={{ fontSize: 12, color: 'var(--muted)', padding: '14px' }}>加载中…</div>
        ) : rows.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--muted)', padding: '24px', textAlign: 'center' }}>
            暂无调用记录 — 到 Playground 跑一次就会出现在这里。
          </div>
        ) : (
          rows.map((t, i) => (
            <UsageRow
              key={t.id}
              task={t}
              outputs={svc.exposed_outputs}
              isFirst={i === 0}
              open={openId === t.id}
              onToggle={() => setOpenId(openId === t.id ? null : t.id)}
            />
          ))
        )}
      </Panel>
    </div>
  )
}

function UsageRow({
  task, outputs, isFirst, open, onToggle,
}: {
  task: ExecutionTask
  outputs: ExposedParam[]
  isFirst: boolean
  open: boolean
  onToggle: () => void
}) {
  const dur = task.duration_ms == null ? '—'
    : task.duration_ms < 1000 ? `${task.duration_ms} ms` : `${(task.duration_ms / 1000).toFixed(2)} s`
  const statusColor = task.status === 'completed' ? 'var(--ok, #34c759)'
    : task.status === 'failed' ? 'var(--error, #ef4444)'
    : 'var(--muted)'
  const inputSummary = task.input_json ? JSON.stringify(task.input_json) : '—'
  return (
    <div style={{ borderTop: isFirst ? 'none' : '1px solid var(--border)' }}>
      <div
        onClick={onToggle}
        style={{
          display: 'grid', gridTemplateColumns: '150px 70px 80px 1fr', gap: 10,
          padding: '9px 14px', fontSize: 12, cursor: 'pointer', alignItems: 'center',
        }}
      >
        <span style={{ color: 'var(--muted)', fontFamily: 'var(--mono, monospace)', fontSize: 11 }}>
          {new Date(task.created_at).toLocaleString()}
        </span>
        <span style={{ color: statusColor }}>{task.status}</span>
        <span style={{ color: 'var(--text)', fontFamily: 'var(--mono, monospace)' }}>{dur}</span>
        <span style={{
          color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }} title={inputSummary}>
          {inputSummary}
        </span>
      </div>
      {open && (
        <div style={{ padding: '4px 14px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>入参</div>
            <pre style={{
              margin: 0, padding: 10, background: 'var(--bg)', border: '1px solid var(--border)',
              borderRadius: 4, fontSize: 11, fontFamily: 'var(--mono, monospace)', color: 'var(--text)', overflow: 'auto',
            }}>
              {JSON.stringify(task.input_json ?? {}, null, 2)}
            </pre>
          </div>
          <div>
            <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>出参</div>
            <SchemaDrivenOutput outputs={outputs} result={task.result} error={task.error} />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------- bits ----------

function StatusBadge({ status }: { status: ServiceStatus }) {
  const map: Record<ServiceStatus, React.CSSProperties> = {
    active: { background: 'rgba(34,197,94,0.15)', color: 'var(--accent-2, #22c55e)' },
    paused: { background: 'rgba(245,158,11,0.15)', color: 'var(--warn, #f59e0b)' },
    deprecated: { background: 'var(--bg)', color: 'var(--muted)', border: '1px solid var(--border)' },
    retired: { background: 'rgba(239,68,68,0.15)', color: 'var(--error, #ef4444)' },
  }
  const labelMap: Record<ServiceStatus, string> = {
    active: '运行中',
    paused: '已暂停',
    deprecated: '已弃用',
    retired: '已下线',
  }
  return (
    <span style={{ fontSize: 11, padding: '3px 9px', borderRadius: 10, ...map[status] }}>
      {labelMap[status]}
    </span>
  )
}

function SourceBadge({ svc }: { svc: ServiceDetailT }) {
  const fromWorkflow = svc.source_type === 'workflow' && !!svc.workflow_id
  if (!fromWorkflow) {
    return (
      <span
        style={{
          fontSize: 11,
          padding: '3px 9px',
          borderRadius: 10,
          background: 'var(--accent-subtle, rgba(99,102,241,0.1))',
          color: 'var(--accent)',
        }}
      >
        快速开通
      </span>
    )
  }
  // 来自 workflow — 显示名字 + 短 ID + 跳转到 workflow 编辑器入口。
  const shortId = svc.workflow_id && svc.workflow_id.length > 6
    ? svc.workflow_id.slice(-6)
    : svc.workflow_id
  return (
    <a
      href={`/workflows/${svc.workflow_id}`}
      title={`workflow_id=${svc.workflow_id}`}
      style={{
        fontSize: 11,
        padding: '3px 9px',
        borderRadius: 10,
        background: 'var(--accent-subtle, rgba(99,102,241,0.1))',
        color: 'var(--accent)',
        textDecoration: 'none',
      }}
    >
      来自 {svc.workflow_name ?? 'Workflow'} #{shortId} · v{svc.version}
    </a>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '10px 14px',
          fontSize: 12,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          borderBottom: '1px solid var(--border)',
        }}
      >
        {title}
      </div>
      <div style={{ padding: '8px 0' }}>{children}</div>
    </div>
  )
}

// ---------- m03 "Key 授权" tab — m10 真实接入 ----------

function RealAuthTab({ serviceId, serviceName }: { serviceId: string; serviceName: string }) {
  const navigate = useNavigate()
  // Snowflake service ID 超过 Number.MAX_SAFE_INTEGER (2^53-1)；
  // 用 Number() 强转会丢精度，URL 变错的 ID → 后端 404 → React Query
  // 永远 isLoading → m03 Key 授权 tab 卡在"加载中..."（实际是接口失败）。
  // 后端的 services 路由把 id 序列化成 str，前端从 URL 拿到也一直是
  // string，路径里直接传 string 给 fetch URL 即可，全程不经 Number()。
  const { data: grants, isLoading } = useServiceGrants(serviceId)
  const { data: allKeys } = useApiKeys()
  const addGrant = useAddGrant()
  const removeGrant = useRemoveGrant()
  const toggleGrant = useToggleGrant()
  const [createOpen, setCreateOpen] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)

  const grantedKeyIds = useMemo(
    () => new Set((grants ?? []).map((g) => g.api_key_id)),
    [grants],
  )
  const candidateKeys = useMemo(
    () => (allKeys ?? []).filter((k) => !grantedKeyIds.has(k.id)),
    [allKeys, grantedKeyIds],
  )

  return (
    <Panel title={`授权 — ${serviceName}`}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 14px',
          gap: 8,
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div style={{ flex: 1, fontSize: 12, color: 'var(--muted)' }}>
          {grants?.length ?? 0} 把 key 已授权访问本服务
        </div>
        <button
          type="button"
          onClick={() => setPickerOpen((p) => !p)}
          style={btnGhost}
        >
          <Plus size={12} style={{ marginRight: 4 }} />
          授权已有 Key
        </button>
        <button
          type="button"
          onClick={() => setCreateOpen(true)}
          style={btnPrimary}
        >
          <Plus size={12} style={{ marginRight: 4 }} />
          新建并授权
        </button>
      </div>

      {pickerOpen && (
        <div style={{ padding: '8px 14px', borderBottom: '1px solid var(--border)' }}>
          {candidateKeys.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>
              所有已存在的 key 都已授权 — 新建一把吧。
            </div>
          ) : (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {candidateKeys.map((k) => (
                <button
                  key={k.id}
                  type="button"
                  onClick={() =>
                    addGrant.mutate(
                      { keyId: k.id, serviceId },
                      { onSuccess: () => setPickerOpen(false) },
                    )
                  }
                  style={pickerChip}
                >
                  {k.label} <span style={{ color: 'var(--muted)' }}>{k.key_prefix}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {isLoading && (
        <div style={{ padding: 14, fontSize: 12, color: 'var(--muted)' }}>加载中…</div>
      )}

      {grants && grants.length === 0 && !isLoading && (
        <div style={{ padding: 24, textAlign: 'center', fontSize: 12, color: 'var(--muted)' }}>
          这个服务还没有任何 key 授权 — 新建或挑一把已有 key 来开通访问。
        </div>
      )}

      {grants && grants.length > 0 && (
        <div>
          {grants.map((g) => {
            const pct = g.pack_total > 0 ? Math.min(100, Math.round((g.pack_used / g.pack_total) * 100)) : 0
            return (
              <div
                key={g.grant_id}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1.5fr 0.8fr 0.7fr',
                  gap: 12,
                  padding: '12px 14px',
                  borderBottom: '1px solid var(--border)',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      fontSize: 13,
                      color: 'var(--text)',
                      cursor: 'pointer',
                    }}
                    onClick={() => navigate(`/api-keys/${g.api_key_id}`)}
                  >
                    <KeyRound size={12} style={{ color: 'var(--muted)' }} />
                    {g.api_key_label}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: 'var(--muted)',
                      fontFamily: 'var(--mono, monospace)',
                      marginTop: 3,
                    }}
                  >
                    {g.api_key_prefix}...
                  </div>
                </div>
                <div>
                  {g.pack_total === 0 ? (
                    <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                      未配额（不限）
                    </span>
                  ) : (
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>
                        {g.pack_used.toLocaleString()} / {g.pack_total.toLocaleString()} ({pct}%)
                      </div>
                      <div
                        style={{
                          height: 6,
                          background: 'var(--bg)',
                          borderRadius: 3,
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: '100%',
                            background: pct > 80 ? '#f87171' : 'var(--accent)',
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
                <div>
                  <span
                    style={{
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 10,
                      background:
                        g.grant_status === 'active'
                          ? 'rgba(34,197,94,0.12)'
                          : 'rgba(248,113,113,0.12)',
                      color: g.grant_status === 'active' ? '#4ade80' : '#f87171',
                    }}
                  >
                    {g.grant_status}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    title={g.grant_status === 'active' ? '暂停' : '恢复'}
                    onClick={() =>
                      toggleGrant.mutate({
                        grantId: g.grant_id,
                        status: g.grant_status === 'active' ? 'paused' : 'active',
                      })
                    }
                    style={iconBtn}
                  >
                    {g.grant_status === 'active' ? <Pause size={12} /> : <Play size={12} />}
                  </button>
                  <button
                    type="button"
                    title="解除授权"
                    onClick={() => removeGrant.mutate(g.grant_id)}
                    style={iconBtn}
                  >
                    <Unlink size={12} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <CreateApiKeyDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        preselectedServiceIds={[serviceId]}
      />
    </Panel>
  )
}

const btnGhost = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 10px',
  fontSize: 11,
  background: 'transparent',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  cursor: 'pointer',
} as const

const btnPrimary = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 10px',
  fontSize: 11,
  background: 'var(--accent)',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
  cursor: 'pointer',
} as const

const pickerChip = {
  fontSize: 11,
  padding: '5px 10px',
  background: 'var(--bg)',
  border: '1px solid var(--border)',
  borderRadius: 14,
  color: 'var(--text)',
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
} as const

const iconBtn = {
  background: 'transparent',
  border: '1px solid var(--border)',
  color: 'var(--text)',
  cursor: 'pointer',
  borderRadius: 4,
  width: 26,
  height: 26,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
} as const

function KV({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div
      style={{
        display: 'flex',
        padding: '6px 14px',
        fontSize: 12,
        gap: 12,
      }}
    >
      <span style={{ color: 'var(--muted)', width: 100, flexShrink: 0 }}>{k}</span>
      <span
        style={{
          color: 'var(--text)',
          fontFamily: mono ? 'var(--mono, monospace)' : undefined,
          wordBreak: 'break-all',
        }}
      >
        {v}
      </span>
    </div>
  )
}

function SmallBtn({
  onClick,
  icon: Icon,
  children,
  danger,
}: {
  onClick: () => void
  icon?: typeof Pause
  children: React.ReactNode
  danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '5px 10px',
        background: 'transparent',
        color: danger ? 'var(--error, #ef4444)' : 'var(--text)',
        border: '1px solid',
        borderColor: danger ? 'var(--error, #ef4444)' : 'var(--border)',
        borderRadius: 4,
        fontSize: 12,
        cursor: 'pointer',
      }}
    >
      {Icon && <Icon size={12} />}
      {children}
    </button>
  )
}

function FullPageMessage({
  message,
  variant,
}: {
  message: string
  variant?: 'error'
}) {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg)',
      }}
    >
      <div
        style={{
          padding: 24,
          maxWidth: 480,
          textAlign: 'center',
          color: variant === 'error' ? 'var(--error, #ef4444)' : 'var(--muted)',
          fontSize: 13,
        }}
      >
        {variant === 'error' && (
          <AlertTriangle size={20} style={{ marginBottom: 8, color: 'var(--error, #ef4444)' }} />
        )}
        {message}
      </div>
    </div>
  )
}
