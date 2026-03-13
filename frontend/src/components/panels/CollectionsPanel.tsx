import FloatingPanel from '../layout/FloatingPanel'

const MOCK_COLLECTIONS = [
  { name: '多角色配音合集', engine: 'CosyVoice2 · 5 presets', tags: ['周杰伦', '新闻女', '+3'] },
  { name: '客服语音包', engine: 'IndexTTS-2 · 2 presets', tags: ['中文客服', 'EN Support'] },
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
            padding: 10,
            transition: 'all 0.12s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'var(--border-strong)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--border)'
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)', marginBottom: 4 }}>
            {c.name}
          </div>
          <div style={{ fontSize: 10, color: 'var(--accent-2)', marginBottom: 3 }}>{c.engine}</div>
          <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, color: 'var(--muted)' }}>
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
