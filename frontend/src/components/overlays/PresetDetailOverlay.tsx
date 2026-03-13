import { Copy, Rocket } from 'lucide-react'
import { usePanelStore } from '../../stores/panel'
import { useVoicePresets } from '../../api/voices'

export default function PresetDetailOverlay() {
  const presetId = usePanelStore((s) => s.selectedPresetId)
  const openApiManagement = usePanelStore((s) => s.openApiManagement)
  const { data: presets } = useVoicePresets()
  const preset = presets?.find((p) => p.id === presetId)

  if (!preset) {
    return (
      <div className="absolute inset-0 overflow-y-auto z-[16]" style={{ background: 'var(--bg)' }}>
        <div style={{ padding: 16, fontSize: 11, color: 'var(--muted)' }}>Preset not found</div>
      </div>
    )
  }

  const params = preset.params as Record<string, unknown>

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
            {preset.engine} · 预设模板
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

            {/* Deploy button */}
            <button
              onClick={() =>
                openApiManagement({ source_type: 'preset', source_id: presetId! })
              }
              className="flex items-center gap-1"
              style={{
                width: '100%',
                fontSize: 11,
                padding: '8px 14px',
                background: 'var(--accent)',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                justifyContent: 'center',
              }}
            >
              <Rocket size={12} /> 部署为服务实例
            </button>
          </div>
        </div>
      </div>
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
