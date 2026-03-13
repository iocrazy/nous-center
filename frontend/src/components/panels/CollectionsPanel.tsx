import FloatingPanel from '../layout/FloatingPanel'

const MOCK_COLLECTIONS = [
  { name: '多角色配音合集', engine: 'CosyVoice2', count: 5, tags: ['周杰伦', '新闻女', '+3'] },
  { name: '客服语音包', engine: 'IndexTTS-2', count: 2, tags: ['中文客服', 'EN Support'] },
]

export default function CollectionsPanel() {
  return (
    <FloatingPanel title="Collections">
      {MOCK_COLLECTIONS.map((c) => (
        <div
          key={c.name}
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
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
              {c.name}
            </span>
            <span style={{ fontSize: 9, color: 'var(--muted)' }}>
              {c.count} presets
            </span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--accent-2)', marginBottom: 4 }}>{c.engine}</div>
          <div className="flex gap-1.5 flex-wrap" style={{ fontSize: 9, color: 'var(--muted)' }}>
            {c.tags.map((tag) => (
              <span
                key={tag}
                style={{ background: 'var(--bg-hover)', padding: '1px 5px', borderRadius: 3 }}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      ))}
    </FloatingPanel>
  )
}
