import { useState } from 'react'
import FloatingPanel from '../layout/FloatingPanel'

const MOCK_WORKFLOWS = [
  { name: '基础合成', nodes: 3, edges: 2, desc: 'Text → TTS → Output' },
  { name: '语音克隆', nodes: 4, edges: 3, desc: 'Text + Ref → TTS → Output' },
  { name: '多段拼接', nodes: 6, edges: 5, desc: '2×(Text → TTS) → Concat → Out' },
]

export default function WorkflowsPanel() {
  const [search, setSearch] = useState('')

  const filtered = MOCK_WORKFLOWS.filter(
    (w) => !search || w.name.includes(search),
  )

  return (
    <FloatingPanel
      title="Workflows"
      searchPlaceholder="Search Workflows..."
      onSearch={setSearch}
    >
      {filtered.map((w) => (
        <PresetCard key={w.name}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)', marginBottom: 4 }}>
            {w.name}
          </div>
          <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, color: 'var(--muted)' }}>
            <Tag>{w.nodes} nodes</Tag>
            <Tag>{w.edges} edges</Tag>
          </div>
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3 }}>{w.desc}</div>
        </PresetCard>
      ))}
    </FloatingPanel>
  )
}

function PresetCard({ children }: { children: React.ReactNode }) {
  return (
    <div
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
    >
      {children}
    </div>
  )
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ background: 'var(--bg-hover)', padding: '1px 5px', borderRadius: 3 }}>
      {children}
    </span>
  )
}
