import { useState } from 'react'
import { usePanelStore } from '../../stores/panel'
import { useSettingsStore } from '../../stores/settings'
import { useVoicePresets } from '../../api/voices'
import {
  usePresetKeys,
  useCreatePresetKey,
  useDeletePresetKey,
  useUpdatePresetStatus,
  type PresetApiKeyCreated,
} from '../../api/presetKeys'

export default function PresetDetailOverlay() {
  const presetId = usePanelStore((s) => s.selectedPresetId)
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl)
  const { data: presets } = useVoicePresets()
  const preset = presets?.find((p) => p.id === presetId)

  const { data: keys, isLoading: keysLoading } = usePresetKeys(presetId)
  const createKey = useCreatePresetKey(presetId ?? '')
  const deleteKey = useDeletePresetKey(presetId ?? '')
  const updateStatus = useUpdatePresetStatus(presetId ?? '')

  const [newKeyLabel, setNewKeyLabel] = useState('')
  const [showNewKeyForm, setShowNewKeyForm] = useState(false)
  const [createdKey, setCreatedKey] = useState<PresetApiKeyCreated | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)

  if (!preset) {
    return (
      <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
        <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>Preset not found</div>
      </div>
    )
  }

  const endpointPath = preset.endpoint_path || `/v1/preset/${preset.id}/synthesize`

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
    updateStatus.mutate(preset.status === 'active' ? 'inactive' : 'active')
  }

  const curlExample = `curl -X POST ${apiBaseUrl}${endpointPath} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <your-api-key>" \\
  -d '{"text": "你好世界"}'`

  const params = preset.params as Record<string, unknown>

  return (
    <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
      <div style={{ padding: 16, maxWidth: 900 }}>
        <div style={{ display: 'flex', gap: 24 }}>
          {/* Left Column */}
          <div style={{ flex: 1 }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)' }}>
                {preset.name}
              </div>
              <button
                onClick={handleToggleStatus}
                disabled={updateStatus.isPending}
                style={{
                  fontSize: 10,
                  padding: '2px 8px',
                  borderRadius: 10,
                  border: 'none',
                  cursor: 'pointer',
                  background: preset.status === 'active' ? '#2a5a3a' : '#5a2a2a',
                  color: preset.status === 'active' ? '#4ade80' : '#f87171',
                }}
              >
                {preset.status}
              </button>
            </div>
            <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 16 }}>
              {preset.engine}
            </div>

            {/* Configuration */}
            <SectionTitle color="var(--accent-2)">CONFIGURATION</SectionTitle>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '90px 1fr',
                gap: '4px 8px',
                fontSize: 11,
                marginBottom: 20,
              }}
            >
              <span style={{ color: 'var(--muted)' }}>engine</span>
              <span style={{ color: 'var(--text-strong)' }}>{preset.engine}</span>
              <span style={{ color: 'var(--muted)' }}>speed</span>
              <span style={{ color: 'var(--text-strong)' }}>{params.speed ?? 1.0}</span>
              <span style={{ color: 'var(--muted)' }}>sample_rate</span>
              <span style={{ color: 'var(--text-strong)' }}>{params.sample_rate ?? 24000}</span>
              <span style={{ color: 'var(--muted)' }}>voice</span>
              <span style={{ color: 'var(--text-strong)' }}>{(params.voice as string) ?? 'default'}</span>
            </div>

            {/* Endpoint */}
            <SectionTitle color="var(--accent-2)">ENDPOINT</SectionTitle>
            <div
              onClick={() => copyToClipboard(`POST ${apiBaseUrl}${endpointPath}`, 'endpoint')}
              style={{
                fontSize: 11,
                fontFamily: 'var(--mono)',
                background: 'var(--card)',
                padding: '6px 10px',
                borderRadius: 4,
                marginBottom: 20,
                cursor: 'pointer',
                color: 'var(--text-strong)',
                border: '1px solid var(--border)',
              }}
            >
              POST {endpointPath}
              <span style={{ fontSize: 9, color: 'var(--muted)', marginLeft: 8 }}>
                {copied === 'endpoint' ? '已复制' : '点击复制'}
              </span>
            </div>

            {/* cURL Example */}
            <SectionTitle color="var(--muted)">CURL EXAMPLE</SectionTitle>
            <pre
              onClick={() => copyToClipboard(curlExample, 'curl')}
              style={{
                fontSize: 10,
                fontFamily: 'var(--mono)',
                background: 'var(--card)',
                padding: '8px 10px',
                borderRadius: 4,
                color: 'var(--text-strong)',
                border: '1px solid var(--border)',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.6,
                cursor: 'pointer',
                margin: 0,
              }}
            >
              {curlExample}
              <span style={{ fontSize: 9, color: 'var(--muted)', display: 'block', marginTop: 4 }}>
                {copied === 'curl' ? '已复制' : '点击复制'}
              </span>
            </pre>
          </div>

          {/* Right Column */}
          <div style={{ flex: 1 }}>
            {/* API Keys Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 12,
              }}
            >
              <SectionTitle color="var(--accent)" style={{ marginBottom: 0 }}>
                API KEYS ({keys?.length ?? 0})
              </SectionTitle>
              <button
                onClick={() => {
                  setShowNewKeyForm(true)
                  setCreatedKey(null)
                }}
                style={{
                  fontSize: 10,
                  padding: '3px 10px',
                  background: 'var(--accent)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                }}
              >
                + New Key
              </button>
            </div>

            {/* Created Key Banner */}
            {createdKey && (
              <div
                style={{
                  background: '#1a2a1a',
                  border: '1px solid #2a5a3a',
                  borderRadius: 6,
                  padding: 10,
                  marginBottom: 10,
                }}
              >
                <div style={{ fontSize: 10, color: '#4ade80', fontWeight: 600, marginBottom: 4 }}>
                  API Key 已创建 — 请立即保存，之后无法再次查看
                </div>
                <div
                  onClick={() => copyToClipboard(createdKey.key, 'created-key')}
                  style={{
                    fontSize: 11,
                    fontFamily: 'var(--mono)',
                    background: 'var(--bg)',
                    padding: '4px 8px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    color: '#4ade80',
                    wordBreak: 'break-all',
                  }}
                >
                  {createdKey.key}
                  <span style={{ fontSize: 9, color: 'var(--muted)', marginLeft: 8 }}>
                    {copied === 'created-key' ? '已复制' : '点击复制'}
                  </span>
                </div>
              </div>
            )}

            {/* New Key Form */}
            {showNewKeyForm && (
              <div
                style={{
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: 10,
                  marginBottom: 10,
                }}
              >
                <label style={{ fontSize: 10, color: 'var(--muted)', display: 'block', marginBottom: 4 }}>
                  Key 标签（如：有声小说App）
                </label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    type="text"
                    value={newKeyLabel}
                    onChange={(e) => setNewKeyLabel(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateKey()}
                    placeholder="输入标签名称"
                    autoFocus
                    style={{
                      flex: 1,
                      padding: '4px 8px',
                      fontSize: 11,
                      fontFamily: 'var(--mono)',
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
                      padding: '4px 12px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: '1px solid var(--accent)',
                      background: 'var(--accent)',
                      color: '#fff',
                      cursor: 'pointer',
                      opacity: createKey.isPending || !newKeyLabel.trim() ? 0.5 : 1,
                    }}
                  >
                    {createKey.isPending ? '...' : '创建'}
                  </button>
                  <button
                    onClick={() => {
                      setShowNewKeyForm(false)
                      setNewKeyLabel('')
                    }}
                    style={{
                      padding: '4px 8px',
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

            {/* Loading */}
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
                  padding: 10,
                  marginBottom: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-strong)' }}>
                    {k.label}
                  </span>
                  {confirmDeleteId === k.id ? (
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button
                        onClick={() => handleDeleteKey(k.id)}
                        style={{
                          fontSize: 10,
                          color: '#f87171',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                        }}
                      >
                        确认删除
                      </button>
                      <button
                        onClick={() => setConfirmDeleteId(null)}
                        style={{
                          fontSize: 10,
                          color: 'var(--muted)',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                        }}
                      >
                        取消
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setConfirmDeleteId(k.id)}
                      style={{
                        fontSize: 10,
                        color: '#f87171',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                      }}
                    >
                      Revoke
                    </button>
                  )}
                </div>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                  {k.key_prefix}...{'  '}·{'  '}
                  {k.usage_calls.toLocaleString()} calls{'  '}·{'  '}
                  {k.usage_chars.toLocaleString()} chars
                </div>
                {k.last_used_at && (
                  <div style={{ fontSize: 10, color: 'var(--muted)' }}>
                    Last used {formatRelativeTime(k.last_used_at)}
                  </div>
                )}
              </div>
            ))}

            {!keysLoading && keys?.length === 0 && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>
                暂无 API Key，点击 "+ New Key" 创建
              </div>
            )}

            {/* Usage Summary */}
            {keys && keys.length > 0 && (
              <>
                <SectionTitle color="var(--accent)" style={{ marginTop: 20 }}>
                  USAGE TOTAL
                </SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: 10,
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 8,
                  }}
                >
                  <div>
                    <div style={{ fontSize: 10, color: 'var(--muted)' }}>Total Calls</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)' }}>
                      {keys.reduce((s, k) => s + k.usage_calls, 0).toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: 'var(--muted)' }}>Total Chars</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)' }}>
                      {keys.reduce((s, k) => s + k.usage_chars, 0).toLocaleString()}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function SectionTitle({
  children,
  color,
  style,
}: {
  children: React.ReactNode
  color: string
  style?: React.CSSProperties
}) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color,
        marginBottom: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
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
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}
