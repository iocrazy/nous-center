import { useState } from 'react'
import FloatingPanel from '../layout/FloatingPanel'
import { useVoicePresets } from '../../api/voices'
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
        <div
          key={p.id}
          className="rounded-md cursor-pointer mb-1.5"
          style={{
            background: 'var(--card)',
            border: '1px solid var(--border)',
            padding: 10,
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
          onClick={() => usePanelStore.getState().openPresetDetail(p.id)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
              {p.name}
            </span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--accent-2)', marginBottom: 3 }}>
            {p.engine}
          </div>
          <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, color: 'var(--muted)' }}>
            {p.tags.map((tag) => (
              <span
                key={tag}
                style={{ background: 'var(--bg-hover)', padding: '1px 5px', borderRadius: 3 }}
              >
                {tag}
              </span>
            ))}
          </div>
          <div style={{ fontSize: 9, marginTop: 4, color: 'var(--ok)' }}>点击查看详情</div>
        </div>
      ))}
    </FloatingPanel>
  )
}
