import { useState } from 'react'
import { Copy, Check, ChevronDown, ChevronUp, Plus, Trash2, Key, Activity } from 'lucide-react'
import { usePanelStore } from '../../stores/panel'
import { useSettingsStore } from '../../stores/settings'
import {
  useInstance,
  useInstanceKeys,
  useCreateInstanceKey,
  useDeleteInstanceKey,
  useUpdateInstanceStatus,
  type InstanceApiKeyCreated,
} from '../../api/instances'

export default function InstanceDetailOverlay() {
  const instanceId = usePanelStore((s) => s.selectedInstanceId)
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl)

  const { data: instance } = useInstance(instanceId)
  const { data: keys, isLoading: keysLoading } = useInstanceKeys(instanceId)
  const createKey = useCreateInstanceKey(instanceId ?? '')
  const deleteKey = useDeleteInstanceKey(instanceId ?? '')
  const updateStatus = useUpdateInstanceStatus(instanceId ?? '')

  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [showNewKeyForm, setShowNewKeyForm] = useState(false)
  const [createdKey, setCreatedKey] = useState<InstanceApiKeyCreated | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [curlExpanded, setCurlExpanded] = useState(false)

  if (!instance) {
    return (
      <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
        <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>Instance not found</div>
      </div>
    )
  }

  const endpointPath = instance.endpoint_path || `/v1/instances/${instance.id}/synthesize`

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    setCopied(label)
    setTimeout(() => setCopied(null), 2000)
  }

  const handleCreateKey = async () => {
    if (!newKeyLabel.trim()) return
    const result = await createKey.mutateAsync(newKeyLabel.trim())
    setCreatedKey(result)
    setNewKeyLabel('')
    setShowNewKeyForm(false)
  }

  const handleDeleteKey = async (keyId: string) => {
    await deleteKey.mutateAsync(keyId)
    setConfirmDeleteId(null)
  }

  const handleToggleStatus = () => {
    updateStatus.mutate(instance.status === 'active' ? 'inactive' : 'active')
  }

  const curlExample = `curl -X POST ${apiBaseUrl}${endpointPath} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <your-api-key>" \\
  -d '{"text": "你好世界"}'`

  const totalCalls = keys?.reduce((s, k) => s + k.usage_calls, 0) ?? 0
  const totalChars = keys?.reduce((s, k) => s + k.usage_chars, 0) ?? 0

  return (
    <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
      <div style={{ padding: '16px 20px', maxWidth: 960 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-strong)', margin: 0 }}>
            {instance.name}
          </h2>
          <button
            onClick={handleToggleStatus}
            disabled={updateStatus.isPending}
            style={{
              fontSize: 10,
              padding: '2px 10px',
              borderRadius: 10,
              border: 'none',
              cursor: 'pointer',
              background: instance.status === 'active' ? 'rgba(34,197,94,0.15)' : 'rgba(248,113,113,0.15)',
              color: instance.status === 'active' ? '#4ade80' : '#f87171',
              fontWeight: 500,
            }}
          >
            {instance.status}
          </button>
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 16 }}>
          {instance.type} · 服务实例
        </div>

        {/* Stats Row */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
          <StatCard icon={<Key size={12} />} label="API Keys" value={String(keys?.length ?? 0)} />
          <StatCard icon={<Activity size={12} />} label="总调用" value={totalCalls.toLocaleString()} />
          <StatCard label="总字符" value={totalChars.toLocaleString()} />
        </div>

        <div style={{ display: 'flex', gap: 20 }}>
          {/* Left Column — Endpoint & Config */}
          <div style={{ width: 360, flexShrink: 0 }}>
            {/* Endpoint */}
            <SectionTitle>Endpoint</SectionTitle>
            <div
              onClick={() => copyToClipboard(`POST ${apiBaseUrl}${endpointPath}`, 'endpoint')}
              className="flex items-center gap-2"
              style={{
                fontSize: 11,
                fontFamily: 'var(--mono)',
                background: 'var(--card)',
                padding: '8px 10px',
                borderRadius: 6,
                marginBottom: 8,
                cursor: 'pointer',
                color: 'var(--text-strong)',
                border: '1px solid var(--border)',
              }}
            >
              <span
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  padding: '1px 5px',
                  borderRadius: 3,
                  background: 'rgba(34,197,94,0.15)',
                  color: 'var(--ok)',
                  flexShrink: 0,
                }}
              >
                POST
              </span>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {endpointPath}
              </span>
              {copied === 'endpoint' ? (
                <Check size={12} style={{ color: 'var(--ok)', flexShrink: 0 }} />
              ) : (
                <Copy size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
              )}
            </div>

            {/* cURL — collapsible */}
            <div
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                marginBottom: 16,
                overflow: 'hidden',
              }}
            >
              <div
                onClick={() => setCurlExpanded(!curlExpanded)}
                className="flex items-center justify-between"
                style={{
                  padding: '6px 10px',
                  cursor: 'pointer',
                  fontSize: 10,
                  color: 'var(--muted)',
                  fontWeight: 500,
                }}
              >
                <span>cURL 示例</span>
                {curlExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </div>
              {curlExpanded && (
                <pre
                  onClick={() => copyToClipboard(curlExample, 'curl')}
                  style={{
                    fontSize: 10,
                    fontFamily: 'var(--mono)',
                    padding: '4px 10px 8px',
                    color: 'var(--text-strong)',
                    whiteSpace: 'pre-wrap',
                    lineHeight: 1.6,
                    cursor: 'pointer',
                    margin: 0,
                    borderTop: '1px solid var(--border)',
                  }}
                >
                  {curlExample}
                  <span style={{ fontSize: 9, color: 'var(--muted)', display: 'block', marginTop: 4 }}>
                    {copied === 'curl' ? '✓ 已复制' : '点击复制'}
                  </span>
                </pre>
              )}
            </div>

            {/* Params Override */}
            {Object.keys(instance.params_override).length > 0 && (
              <>
                <SectionTitle>参数覆盖</SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '6px 0',
                  }}
                >
                  {Object.entries(instance.params_override).map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '3px 10px',
                        fontSize: 11,
                      }}
                    >
                      <span style={{ color: 'var(--muted)' }}>{k}</span>
                      <span style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>
                        {String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Right Column — API Keys */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 10,
              }}
            >
              <SectionTitle style={{ marginBottom: 0 }}>
                API Keys ({keys?.length ?? 0})
              </SectionTitle>
              <button
                onClick={() => { setShowNewKeyForm(true); setCreatedKey(null) }}
                className="flex items-center gap-1"
                style={{
                  fontSize: 10,
                  padding: '4px 10px',
                  background: 'var(--accent)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                }}
              >
                <Plus size={10} /> New Key
              </button>
            </div>

            {/* Created Key Banner */}
            {createdKey && (
              <div
                style={{
                  background: 'rgba(34,197,94,0.08)',
                  border: '1px solid rgba(34,197,94,0.3)',
                  borderRadius: 6,
                  padding: 10,
                  marginBottom: 10,
                }}
              >
                <div style={{ fontSize: 10, color: '#4ade80', fontWeight: 600, marginBottom: 6 }}>
                  ⚠ 请立即保存此 Key，关闭后无法再次查看
                </div>
                <div
                  onClick={() => copyToClipboard(createdKey.key, 'created-key')}
                  className="flex items-center gap-2"
                  style={{
                    fontSize: 11,
                    fontFamily: 'var(--mono)',
                    background: 'var(--bg)',
                    padding: '6px 8px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    color: '#4ade80',
                    wordBreak: 'break-all',
                  }}
                >
                  <span style={{ flex: 1 }}>{createdKey.key}</span>
                  {copied === 'created-key' ? (
                    <Check size={12} style={{ flexShrink: 0 }} />
                  ) : (
                    <Copy size={12} style={{ color: 'var(--muted)', flexShrink: 0 }} />
                  )}
                </div>
              </div>
            )}

            {/* New Key Form */}
            {showNewKeyForm && (
              <div
                style={{
                  background: 'var(--card)',
                  border: '1px solid var(--accent)',
                  borderRadius: 6,
                  padding: 10,
                  marginBottom: 10,
                }}
              >
                <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4 }}>
                  Key 标签
                </label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    type="text"
                    value={newKeyLabel}
                    onChange={(e) => setNewKeyLabel(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateKey()}
                    placeholder="如：有声小说App"
                    autoFocus
                    style={{
                      flex: 1,
                      padding: '5px 8px',
                      fontSize: 11,
                      background: 'var(--bg)',
                      border: '1px solid var(--border)',
                      borderRadius: 4,
                      color: 'var(--text-strong)',
                      outline: 'none',
                    }}
                  />
                  <button
                    onClick={handleCreateKey}
                    disabled={createKey.isPending || !newKeyLabel.trim()}
                    style={{
                      padding: '5px 14px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: 'none',
                      background: 'var(--accent)',
                      color: '#fff',
                      cursor: 'pointer',
                      opacity: createKey.isPending || !newKeyLabel.trim() ? 0.5 : 1,
                    }}
                  >
                    {createKey.isPending ? '...' : '创建'}
                  </button>
                  <button
                    onClick={() => { setShowNewKeyForm(false); setNewKeyLabel('') }}
                    style={{
                      padding: '5px 8px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: '1px solid var(--border)',
                      background: 'none',
                      color: 'var(--muted)',
                      cursor: 'pointer',
                    }}
                  >
                    取消
                  </button>
                </div>
              </div>
            )}

            {keysLoading && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
            )}

            {/* Key List */}
            {keys?.map((k) => (
              <div
                key={k.id}
                style={{
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '8px 10px',
                  marginBottom: 6,
                  transition: 'border-color 0.12s',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-strong)' }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div className="flex items-center gap-2">
                    <Key size={11} style={{ color: 'var(--muted)' }} />
                    <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-strong)' }}>
                      {k.label}
                    </span>
                  </div>
                  {confirmDeleteId === k.id ? (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleDeleteKey(k.id)}
                        style={{ fontSize: 10, color: '#f87171', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 500 }}
                      >
                        确认
                      </button>
                      <button
                        onClick={() => setConfirmDeleteId(null)}
                        style={{ fontSize: 10, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer' }}
                      >
                        取消
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmDeleteId(k.id)}
                      className="flex items-center gap-1"
                      style={{ fontSize: 10, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer' }}
                    >
                      <Trash2 size={10} /> 撤销
                    </button>
                  )}
                </div>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3, fontFamily: 'var(--mono)' }}>
                  {k.key_prefix}...
                  <span style={{ fontFamily: 'var(--font)', marginLeft: 8 }}>
                    {k.usage_calls.toLocaleString()} 次调用 · {k.usage_chars.toLocaleString()} 字符
                  </span>
                  {k.last_used_at && (
                    <span style={{ marginLeft: 8 }}>
                      · {formatRelativeTime(k.last_used_at)}
                    </span>
                  )}
                </div>
              </div>
            ))}

            {!keysLoading && keys?.length === 0 && !showNewKeyForm && (
              <div
                style={{
                  padding: '20px 16px',
                  textAlign: 'center',
                  border: '1px dashed var(--border)',
                  borderRadius: 6,
                  color: 'var(--muted)',
                  fontSize: 11,
                }}
              >
                暂无 API Key — 创建一个用于鉴权访问
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value }: { icon?: React.ReactNode; label: string; value: string }) {
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '8px 12px',
        flex: 1,
        minWidth: 0,
      }}
    >
      <div className="flex items-center gap-1.5" style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>
        {icon}
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)' }}>
        {value}
      </div>
    </div>
  )
}

function SectionTitle({
  children,
  style,
}: {
  children: React.ReactNode
  style?: React.CSSProperties
}) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: 'var(--muted)',
        marginBottom: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return '刚刚'
  if (mins < 60) return `${mins}分钟前`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}小时前`
  const days = Math.floor(hours / 24)
  return `${days}天前`
}
