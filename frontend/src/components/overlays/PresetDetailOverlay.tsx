import { useState } from 'react'
import { Copy, Plus, ChevronRight } from 'lucide-react'
import { usePanelStore } from '../../stores/panel'
import { useVoicePresets } from '../../api/voices'
import { useInstances, useCreateInstance, type ServiceInstance } from '../../api/instances'

export default function PresetDetailOverlay() {
  const presetId = usePanelStore((s) => s.selectedPresetId)
  const { data: presets } = useVoicePresets()
  const preset = presets?.find((p) => p.id === presetId)
  const { data: instances, isLoading: instancesLoading } = useInstances(presetId)
  const createInstance = useCreateInstance()

  const [showNewForm, setShowNewForm] = useState(false)
  const [newName, setNewName] = useState('')

  if (!preset) {
    return (
      <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
        <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>Preset not found</div>
      </div>
    )
  }

  const handleCreateInstance = async () => {
    if (!newName.trim() || !presetId) return
    await createInstance.mutateAsync({ preset_id: presetId, name: newName.trim() })
    setNewName('')
    setShowNewForm(false)
  }

  const params = preset.params as Record<string, unknown>
  const activeInstances = instances?.filter((i) => i.status === 'active').length ?? 0

  return (
    <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
      <div style={{ padding: '16px 20px', maxWidth: 960 }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-strong)', margin: 0 }}>
              {preset.name}
            </h2>
            {preset.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  fontSize: 9,
                  padding: '2px 7px',
                  borderRadius: 3,
                  background: 'var(--bg-hover)',
                  color: 'var(--muted)',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            {preset.engine} · 预设模板 · {instances?.length ?? 0} 个实例（{activeInstances} 活跃）
          </div>
        </div>

        <div style={{ display: 'flex', gap: 20 }}>
          {/* Left Column — Configuration */}
          <div style={{ width: 280, flexShrink: 0 }}>
            <SectionTitle>配置参数</SectionTitle>
            <div
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '8px 0',
                marginBottom: 16,
              }}
            >
              <ConfigRow label="engine" value={preset.engine} />
              {Object.entries(params).map(([k, v]) => (
                <ConfigRow key={k} label={k} value={String(v)} />
              ))}
            </div>

            {preset.reference_audio_path && (
              <>
                <SectionTitle>参考音频</SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 10,
                    color: 'var(--text-strong)',
                    fontFamily: 'var(--mono)',
                    wordBreak: 'break-all',
                    marginBottom: 16,
                  }}
                >
                  {preset.reference_audio_path}
                </div>
              </>
            )}

            {preset.reference_text && (
              <>
                <SectionTitle>参考文本</SectionTitle>
                <div
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontSize: 10,
                    color: 'var(--text-strong)',
                    lineHeight: 1.5,
                    marginBottom: 16,
                  }}
                >
                  {preset.reference_text}
                </div>
              </>
            )}
          </div>

          {/* Right Column — Instances */}
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
                服务实例 ({instances?.length ?? 0})
              </SectionTitle>
              <button
                onClick={() => setShowNewForm(true)}
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
                <Plus size={10} /> 新建实例
              </button>
            </div>

            {/* New Instance Form */}
            {showNewForm && (
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
                  实例名称
                </label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateInstance()}
                    placeholder="如：有声小说频道、播客App"
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
                    onClick={handleCreateInstance}
                    disabled={createInstance.isPending || !newName.trim()}
                    style={{
                      padding: '5px 14px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: 'none',
                      background: 'var(--accent)',
                      color: '#fff',
                      cursor: 'pointer',
                      opacity: createInstance.isPending || !newName.trim() ? 0.5 : 1,
                    }}
                  >
                    {createInstance.isPending ? '...' : '创建'}
                  </button>
                  <button
                    onClick={() => { setShowNewForm(false); setNewName('') }}
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

            {instancesLoading && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
            )}

            {instances?.map((inst) => (
              <InstanceCard key={inst.id} instance={inst} />
            ))}

            {!instancesLoading && instances?.length === 0 && !showNewForm && (
              <div
                style={{
                  padding: '24px 16px',
                  textAlign: 'center',
                  border: '1px dashed var(--border)',
                  borderRadius: 6,
                  color: 'var(--muted)',
                  fontSize: 11,
                }}
              >
                暂无实例 — 创建一个实例来开放 API 服务
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function InstanceCard({ instance }: { instance: ServiceInstance }) {
  const openInstanceDetail = usePanelStore((s) => s.openInstanceDetail)

  return (
    <div
      onClick={() => openInstanceDetail(instance.id)}
      className="rounded-md cursor-pointer flex items-center"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '10px 12px',
        marginBottom: 6,
        transition: 'all 0.12s',
        gap: 10,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-strong)'
        e.currentTarget.style.background = 'var(--bg-elevated)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.background = 'var(--card)'
      }}
    >
      {/* Status dot */}
      <div
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: instance.status === 'active' ? '#4ade80' : '#f87171',
          flexShrink: 0,
        }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-strong)' }}>
          {instance.name}
        </div>
        <div
          style={{
            fontSize: 10,
            color: 'var(--muted)',
            fontFamily: 'var(--mono)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {instance.endpoint_path ?? '—'}
        </div>
      </div>
      <ChevronRight size={14} style={{ color: 'var(--muted)', flexShrink: 0 }} />
    </div>
  )
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '3px 10px',
        fontSize: 11,
      }}
    >
      <span style={{ color: 'var(--muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-strong)', fontFamily: 'var(--mono)' }}>{value}</span>
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
