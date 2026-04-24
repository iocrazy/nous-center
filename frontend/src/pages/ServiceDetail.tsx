import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Code2,
  KeyRound,
  LayoutGrid,
  Pause,
  Play,
  Plus,
  Trash2,
  Unlink,
} from 'lucide-react'
import {
  endpointFor,
  useDeleteService,
  usePatchService,
  useService,
  type ExposedParam,
  type ServiceDetail as ServiceDetailT,
  type ServiceStatus,
} from '../api/services'
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
import { apiFetch } from '../api/client'

export interface ServiceDetailPageProps {
  serviceId: string
  onBack?: () => void
}

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
        {tab === 'playground' && <PlaygroundTab svc={svc} />}
        {tab === 'docs' && <DocsTab svc={svc} />}
        {tab === 'auth' && <AuthTab svc={svc} />}
        {tab === 'usage' && <UsageTab svc={svc} />}
      </div>
    </div>
  )
}

function Breadcrumb({ name, onBack }: { name: string; onBack?: () => void }) {
  return (
    <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>
      <button
        type="button"
        onClick={onBack}
        style={{
          background: 'transparent',
          border: 'none',
          color: 'var(--muted)',
          cursor: 'pointer',
          padding: 0,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
        }}
      >
        <ArrowLeft size={12} /> 服务
      </button>
      <span style={{ margin: '0 6px' }}>/</span>
      <span style={{ color: 'var(--text)' }}>{name}</span>
    </div>
  )
}

function Header({ svc }: { svc: ServiceDetailT }) {
  const patch = usePatchService()
  const del = useDeleteService()

  const setStatus = (s: ServiceStatus) =>
    patch.mutate({ serviceId: svc.id, status: s })

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
          <h1 style={{ fontSize: 22, color: 'var(--text)' }}>{svc.name}</h1>
          <StatusBadge status={svc.status} />
          <SourceBadge svc={svc} />
        </div>
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
          onClick={() => {
            if (confirm(`下线服务 ${svc.name}？此操作不可逆`)) {
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
  )
}

function PlaygroundTab({ svc }: { svc: ServiceDetailT }) {
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [taskInfo, setTaskInfo] = useState<{
    task_id?: string
    status?: 'running' | 'completed' | 'failed'
    latency_ms?: number
  } | null>(null)

  const submit = async (values: Record<string, unknown>) => {
    setRunning(true)
    setError(null)
    setResult(null)
    setTaskInfo({ status: 'running' })
    const start = performance.now()
    try {
      const url =
        svc.category === 'llm'
          ? '/v1/chat/completions'
          : `/v1/apps/${encodeURIComponent(svc.name)}/run`
      const body = svc.category === 'llm' ? buildLlmBody(svc, values) : values
      const data = await apiFetch<Record<string, unknown>>(url, {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setResult(data)
      setTaskInfo({
        status: 'completed',
        latency_ms: Math.round(performance.now() - start),
      })
    } catch (e) {
      const msg = (e as Error).message ?? String(e)
      setError(msg)
      setTaskInfo({
        status: 'failed',
        latency_ms: Math.round(performance.now() - start),
      })
    } finally {
      setRunning(false)
    }
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 360px',
        gap: 14,
        minHeight: 540,
      }}
    >
      <div
        style={{
          background: 'var(--bg-accent)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <SchemaDrivenForm
          inputs={svc.exposed_inputs}
          submitting={running}
          onSubmit={submit}
          estimateLine={`endpoint：${endpointFor(svc)}`}
        />
      </div>
      <SchemaDrivenOutput
        outputs={svc.exposed_outputs}
        result={result}
        taskInfo={taskInfo ?? undefined}
        error={error}
      />
    </div>
  )
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
  return (
    <Panel title={`用量 — ${svc.name}`}>
      <div style={{ fontSize: 12, color: 'var(--muted)', padding: '8px 12px' }}>
        细分图表与统计将随用量子系统接入。当前可在右侧 Usage 概览页查看汇总。
      </div>
    </Panel>
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
  icon: typeof Pause
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
      <Icon size={12} />
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
