import { useState } from 'react'
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

  return (
    <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
      <div style={{ padding: 16, maxWidth: 900 }}>
        <div style={{ display: 'flex', gap: 24 }}>
          {/* Left Column — Preset Config */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-strong)', marginBottom: 4 }}>
              {preset.name}
            </div>
            <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 16 }}>
              {preset.engine} · 预设模板
            </div>

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

            {preset.reference_audio_path && (
              <>
                <SectionTitle color="var(--accent-2)">REFERENCE AUDIO</SectionTitle>
                <div style={{ fontSize: 11, color: 'var(--text-strong)', marginBottom: 20 }}>
                  {preset.reference_audio_path}
                </div>
              </>
            )}

            {preset.tags.length > 0 && (
              <>
                <SectionTitle color="var(--accent-2)">TAGS</SectionTitle>
                <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, marginBottom: 20 }}>
                  {preset.tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        background: 'var(--bg-hover)',
                        padding: '2px 8px',
                        borderRadius: 4,
                        color: 'var(--muted)',
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Right Column — Instances */}
          <div style={{ flex: 1 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 12,
              }}
            >
              <SectionTitle color="var(--accent)" style={{ marginBottom: 0 }}>
                INSTANCES ({instances?.length ?? 0})
              </SectionTitle>
              <button
                onClick={() => setShowNewForm(true)}
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
                + New Instance
              </button>
            </div>

            {/* New Instance Form */}
            {showNewForm && (
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
                  实例名称（如：有声小说频道）
                </label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateInstance()}
                    placeholder="输入实例名称"
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
                    onClick={handleCreateInstance}
                    disabled={createInstance.isPending || !newName.trim()}
                    style={{
                      padding: '4px 12px',
                      fontSize: 10,
                      borderRadius: 4,
                      border: '1px solid var(--accent)',
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

            {instancesLoading && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
            )}

            {instances?.map((inst) => (
              <InstanceCard key={inst.id} instance={inst} />
            ))}

            {!instancesLoading && instances?.length === 0 && (
              <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>
                暂无实例，点击 "+ New Instance" 创建
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
      className="rounded-md cursor-pointer"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: 10,
        marginBottom: 8,
        transition: 'all 0.12s',
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
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-strong)' }}>
          {instance.name}
        </span>
        <span
          style={{
            fontSize: 9,
            padding: '1px 5px',
            borderRadius: 8,
            background: instance.status === 'active' ? '#2a5a3a' : '#5a2a2a',
            color: instance.status === 'active' ? '#4ade80' : '#f87171',
          }}
        >
          {instance.status}
        </span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--muted)' }}>
        {instance.endpoint_path ?? '—'}
      </div>
      <div style={{ fontSize: 9, marginTop: 4, color: 'var(--ok)' }}>点击管理</div>
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
