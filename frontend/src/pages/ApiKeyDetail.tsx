import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Eye,
  EyeOff,
  Pause,
  Play,
  RotateCw,
  Trash2,
  Unlink,
} from 'lucide-react'
import {
  endpointsFor,
  useApiKey,
  useDeleteApiKey,
  usePatchApiKey,
  useRemoveGrant,
  useResetApiKey,
  useToggleGrant,
  type ApiKeyCreated,
  type GrantSummary,
} from '../api/keys'
import { useSettingsStore } from '../stores/settings'

export default function ApiKeyDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl)
  const { data: key, isLoading, error } = useApiKey(id ?? null)
  const reset = useResetApiKey()
  const patch = usePatchApiKey()
  const remove = useDeleteApiKey()
  const toggleGrant = useToggleGrant()
  const removeGrant = useRemoveGrant()

  const [showSecret, setShowSecret] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [resetResult, setResetResult] = useState<ApiKeyCreated | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [expandedSnippet, setExpandedSnippet] = useState<string | null>(null)

  const sampleService = useMemo(() => {
    return key?.grants.find((g) => g.status === 'active') ?? key?.grants[0]
  }, [key])

  const endpoints = useMemo(() => {
    return endpointsFor(sampleService?.service_name ?? '<service>', apiBaseUrl)
  }, [sampleService, apiBaseUrl])

  const handleCopy = async (text: string, tag: string) => {
    await navigator.clipboard.writeText(text)
    setCopied(tag)
    setTimeout(() => setCopied(null), 1500)
  }

  const handleReset = async () => {
    if (!id) return
    try {
      const out = await reset.mutateAsync(id)
      setResetResult(out)
      setShowSecret(true)
    } finally {
      setConfirmReset(false)
    }
  }

  const handleDelete = async () => {
    if (!id) return
    try {
      await remove.mutateAsync(id)
      navigate('/api-keys')
    } catch {
      setConfirmDelete(false)
    }
  }

  if (isLoading || !key) {
    return (
      <div style={{ padding: 36, color: 'var(--muted)', fontSize: 13 }}>
        {error ? `加载失败：${(error as Error).message}` : '加载中...'}
      </div>
    )
  }

  const visibleSecret: string | null = showSecret
    ? (resetResult?.secret ?? key.secret_plaintext ?? null)
    : null
  const secretDisplay = visibleSecret ?? `${key.key_prefix}${'•'.repeat(20)}`

  const sampleSecret = visibleSecret ?? key.secret_plaintext ?? '<your-api-key>'
  const sampleModel = sampleService?.service_name ?? '<service>'

  const snippets = buildSnippets(apiBaseUrl, sampleModel, sampleSecret)

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'auto',
        background: 'var(--bg)',
      }}
    >
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
        {/* breadcrumb */}
        <button
          type="button"
          onClick={() => navigate('/api-keys')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: 0,
            background: 'transparent',
            color: 'var(--muted)',
            border: 'none',
            cursor: 'pointer',
            fontSize: 12,
            marginBottom: 12,
          }}
        >
          <ArrowLeft size={12} /> 返回 API Key 列表
        </button>

        {/* header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', marginBottom: 18 }}>
          <div style={{ flex: 1 }}>
            <h1 style={{ fontSize: 22, fontWeight: 600, color: 'var(--text)' }}>
              {key.label}
            </h1>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>
              ID #{key.id} · 创建于 {fmtDate(key.created_at)}{' '}
              {key.expires_at ? `· 过期 ${fmtDate(key.expires_at)}` : ''}
            </div>
            {key.note && (
              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>
                {key.note}
              </div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              onClick={() =>
                patch.mutate({
                  keyId: key.id,
                  body: { is_active: !key.is_active },
                })
              }
              style={btnGhost}
            >
              {key.is_active ? (
                <>
                  <Pause size={12} style={{ marginRight: 4 }} />
                  停用
                </>
              ) : (
                <>
                  <Play size={12} style={{ marginRight: 4 }} />
                  启用
                </>
              )}
            </button>
            <button
              type="button"
              onClick={() => setConfirmReset(true)}
              style={{ ...btnGhost, color: 'var(--accent)', borderColor: 'var(--accent)' }}
            >
              <RotateCw size={12} style={{ marginRight: 4 }} />
              Reset
            </button>
            <button
              type="button"
              onClick={() => setConfirmDelete(true)}
              style={{ ...btnGhost, color: '#f87171', borderColor: '#f87171' }}
            >
              <Trash2 size={12} style={{ marginRight: 4 }} />
              删除
            </button>
          </div>
        </div>

        {/* secret panel */}
        <Section title="Secret">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: 'var(--bg-accent)',
              padding: '10px 12px',
              borderRadius: 6,
              border: '1px solid var(--border)',
            }}
          >
            <span
              style={{
                fontFamily: 'var(--mono, monospace)',
                fontSize: 13,
                color: visibleSecret ? '#4ade80' : 'var(--text)',
                flex: 1,
                wordBreak: 'break-all',
              }}
            >
              {secretDisplay}
            </span>
            <IconBtn
              title={showSecret ? '隐藏' : '查看明文'}
              onClick={() => setShowSecret((p) => !p)}
              disabled={!key.secret_plaintext && !resetResult}
            >
              {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
            </IconBtn>
            <IconBtn
              title="复制"
              onClick={() => visibleSecret && handleCopy(visibleSecret, 'secret')}
              disabled={!visibleSecret}
            >
              {copied === 'secret' ? <Check size={14} /> : <Copy size={14} />}
            </IconBtn>
          </div>
          {!key.secret_plaintext && !resetResult && (
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
              这是一把旧 key — 创建时未保存明文。点 Reset 一次即可获得可常驻显示的新 secret。
            </div>
          )}
        </Section>

        {/* endpoints */}
        <Section title="调用方式">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: 10,
            }}
          >
            {(['openai', 'ollama', 'anthropic'] as const).map((proto) => {
              const ep = endpoints[proto]
              return (
                <div
                  key={proto}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: 12,
                    background: 'var(--bg-accent)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: 6,
                    }}
                  >
                    <span style={{ fontSize: 12, color: 'var(--text)', fontWeight: 500 }}>
                      {ep.label}
                    </span>
                    <IconBtn
                      title="复制 endpoint"
                      onClick={() => handleCopy(ep.url, `ep-${proto}`)}
                    >
                      {copied === `ep-${proto}` ? <Check size={12} /> : <Copy size={12} />}
                    </IconBtn>
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--mono, monospace)',
                      fontSize: 11,
                      color: 'var(--muted)',
                      wordBreak: 'break-all',
                    }}
                  >
                    {ep.url}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
                    {ep.hint}
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setExpandedSnippet((p) => (p === proto ? null : proto))
                    }
                    style={{
                      marginTop: 8,
                      padding: '4px 8px',
                      background: 'transparent',
                      color: 'var(--muted)',
                      border: '1px solid var(--border)',
                      borderRadius: 4,
                      fontSize: 11,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                  >
                    {expandedSnippet === proto ? (
                      <ChevronUp size={11} />
                    ) : (
                      <ChevronDown size={11} />
                    )}
                    cURL
                  </button>
                  {expandedSnippet === proto && (
                    <pre
                      onClick={() => handleCopy(snippets[proto], `snip-${proto}`)}
                      style={{
                        marginTop: 8,
                        padding: 10,
                        background: 'var(--bg)',
                        border: '1px solid var(--border)',
                        borderRadius: 4,
                        fontFamily: 'var(--mono, monospace)',
                        fontSize: 10,
                        color: 'var(--text)',
                        whiteSpace: 'pre-wrap',
                        cursor: 'pointer',
                        margin: '8px 0 0',
                        lineHeight: 1.5,
                      }}
                    >
                      {snippets[proto]}
                      {copied === `snip-${proto}` && (
                        <span
                          style={{
                            display: 'block',
                            marginTop: 6,
                            color: '#4ade80',
                            fontSize: 9,
                          }}
                        >
                          已复制
                        </span>
                      )}
                    </pre>
                  )}
                </div>
              )
            })}
          </div>
        </Section>

        {/* grants */}
        <Section title={`授权服务 (${key.grants.length})`}>
          {key.grants.length === 0 ? (
            <div
              style={{
                padding: 24,
                textAlign: 'center',
                border: '1px dashed var(--border)',
                borderRadius: 6,
                color: 'var(--muted)',
                fontSize: 12,
              }}
            >
              这把 key 还没授权任何服务。
              <br />
              到服务详情页 → "Key 授权" tab 把它加进去。
            </div>
          ) : (
            <div
              style={{
                border: '1px solid var(--border)',
                borderRadius: 6,
                overflow: 'hidden',
              }}
            >
              {key.grants.map((g) => (
                <GrantRow
                  key={g.id}
                  g={g}
                  onToggle={() =>
                    toggleGrant.mutate({
                      grantId: g.id,
                      status: g.status === 'active' ? 'paused' : 'active',
                    })
                  }
                  onRemove={() => removeGrant.mutate(g.id)}
                />
              ))}
            </div>
          )}
        </Section>

        {/* usage */}
        <Section title="用量">
          <div style={{ display: 'flex', gap: 12 }}>
            <Stat label="总调用" value={key.usage_calls.toLocaleString()} />
            <Stat label="字符" value={key.usage_chars.toLocaleString()} />
            <Stat label="最近使用" value={fmtDate(key.last_used_at) ?? '—'} />
          </div>
        </Section>
      </div>

      {confirmReset && (
        <ConfirmDialog
          title="重置 Secret？"
          description="旧 secret 会立即失效（所有用旧 key 的客户端都会 401）。授权关系不变。"
          danger="重置"
          loading={reset.isPending}
          onCancel={() => setConfirmReset(false)}
          onConfirm={handleReset}
        />
      )}
      {confirmDelete && (
        <ConfirmDialog
          title="删除 API Key？"
          description="不可撤销。所有授权同时被回收。"
          danger="删除"
          loading={remove.isPending}
          onCancel={() => setConfirmDelete(false)}
          onConfirm={handleDelete}
        />
      )}
    </div>
  )
}

// ---------- subviews ----------

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <h2
        style={{
          fontSize: 12,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.6,
          fontWeight: 600,
          marginBottom: 8,
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  )
}

function GrantRow({
  g,
  onToggle,
  onRemove,
}: {
  g: GrantSummary
  onToggle: () => void
  onRemove: () => void
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.6fr 1fr 0.8fr 0.7fr',
        gap: 12,
        padding: '10px 14px',
        borderBottom: '1px solid var(--border)',
        alignItems: 'center',
      }}
    >
      <div>
        <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>
          {g.service_name}
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
          {g.service_category ?? '—'} · grant #{g.id}
        </div>
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)' }}>
        授权于 {fmtDate(g.activated_at) ?? '—'}
      </div>
      <div>
        <span
          style={{
            fontSize: 11,
            padding: '2px 8px',
            borderRadius: 10,
            background:
              g.status === 'active'
                ? 'rgba(34,197,94,0.12)'
                : 'rgba(248,113,113,0.12)',
            color: g.status === 'active' ? '#4ade80' : '#f87171',
          }}
        >
          {g.status}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
        <IconBtn title={g.status === 'active' ? '暂停' : '恢复'} onClick={onToggle}>
          {g.status === 'active' ? <Pause size={12} /> : <Play size={12} />}
        </IconBtn>
        <IconBtn title="解除授权" onClick={onRemove}>
          <Unlink size={12} />
        </IconBtn>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        flex: 1,
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '12px 14px',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--muted)' }}>{label}</div>
      <div
        style={{
          fontSize: 18,
          color: 'var(--text)',
          fontWeight: 600,
          marginTop: 4,
        }}
      >
        {value}
      </div>
    </div>
  )
}

function IconBtn({
  title,
  onClick,
  disabled,
  children,
}: {
  title: string
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={(e) => {
        e.stopPropagation()
        if (!disabled) onClick()
      }}
      disabled={disabled}
      style={{
        background: 'transparent',
        border: '1px solid var(--border)',
        color: disabled ? 'var(--muted)' : 'var(--text)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        borderRadius: 4,
        width: 26,
        height: 26,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {children}
    </button>
  )
}

function ConfirmDialog({
  title,
  description,
  danger,
  loading,
  onCancel,
  onConfirm,
}: {
  title: string
  description: string
  danger: string
  loading?: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.55)',
        zIndex: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-elevated, var(--bg))',
          border: '1px solid var(--border)',
          borderRadius: 8,
          width: 380,
          padding: 20,
          boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
        }}
      >
        <h3 style={{ fontSize: 14, color: 'var(--text)', fontWeight: 600 }}>
          {title}
        </h3>
        <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8, lineHeight: 1.6 }}>
          {description}
        </p>
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            marginTop: 16,
          }}
        >
          <button onClick={onCancel} type="button" style={btnGhost}>
            取消
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            type="button"
            style={{
              ...btnGhost,
              background: '#ef4444',
              color: '#fff',
              borderColor: '#ef4444',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? '...' : danger}
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------- helpers ----------

function fmtDate(iso: string | null): string | null {
  if (!iso) return null
  const d = new Date(iso)
  if (isNaN(d.getTime())) return null
  return d.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function buildSnippets(
  baseUrl: string,
  model: string,
  apiKey: string,
): Record<'openai' | 'ollama' | 'anthropic', string> {
  return {
    openai: `curl ${baseUrl}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{
    "model": "${model}",
    "messages": [{"role":"user","content":"hello"}]
  }'`,
    ollama: `curl ${baseUrl}/api/chat \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{
    "model": "${model}",
    "messages": [{"role":"user","content":"hello"}]
  }'`,
    anthropic: `curl ${baseUrl}/v1/messages \\
  -H "x-api-key: ${apiKey}" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "${model}",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"hello"}]
  }'`,
  }
}

const btnGhost = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 12px',
  fontSize: 12,
  background: 'transparent',
  color: 'var(--muted)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  cursor: 'pointer',
} as const
