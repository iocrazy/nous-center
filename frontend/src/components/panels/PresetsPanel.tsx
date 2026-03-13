import { useState } from 'react'
import FloatingPanel from '../layout/FloatingPanel'
import { useVoicePresets } from '../../api/voices'
import { useInstances } from '../../api/instances'
import { usePanelStore } from '../../stores/panel'

export default function PresetsPanel() {
  const [search, setSearch] = useState('')
  const { data: presets, isLoading } = useVoicePresets()

  const filtered = (presets ?? []).filter(
    (p) => !search || p.name.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <FloatingPanel
      title="Presets"
      searchPlaceholder="Search Presets..."
      onSearch={setSearch}
    >
      {isLoading && (
        <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>
          {search ? '无匹配结果' : '暂无预设'}
        </div>
      )}

      {filtered.map((p) => (
        <PresetCard key={p.id} preset={p} />
      ))}
    </FloatingPanel>
  )
}

function PresetCard({ preset }: { preset: { id: string; name: string; engine: string; tags: string[] } }) {
  const { data: instances } = useInstances(preset.id)
  const instanceCount = instances?.length ?? 0

  return (
    <div
      className="rounded-md cursor-pointer mb-1.5"
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: '8px 10px',
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
      onClick={() => usePanelStore.getState().openPresetDetail(preset.id)}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
          {preset.name}
        </span>
        {instanceCount > 0 && (
          <span
            style={{
              fontSize: 9,
              padding: '1px 6px',
              borderRadius: 8,
              background: 'rgba(34,197,94,0.12)',
              color: 'var(--ok)',
              fontWeight: 500,
            }}
          >
            {instanceCount} 实例
          </span>
        )}
      </div>
      <div style={{ fontSize: 10, color: 'var(--accent-2)', marginBottom: 4 }}>
        {preset.engine}
      </div>
      {preset.tags.length > 0 && (
        <div className="flex gap-1.5 flex-wrap" style={{ fontSize: 9, color: 'var(--muted)' }}>
          {preset.tags.map((tag) => (
            <span
              key={tag}
              style={{ background: 'var(--bg-hover)', padding: '1px 5px', borderRadius: 3 }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
