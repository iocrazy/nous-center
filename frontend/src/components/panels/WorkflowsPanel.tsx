import { useState } from 'react'
import FloatingPanel from '../layout/FloatingPanel'
import { useWorkflows, useCreateWorkflow, useDeleteWorkflow } from '../../api/workflows'
import { useWorkspaceStore } from '../../stores/workspace'

export default function WorkflowsPanel() {
  const [search, setSearch] = useState('')
  const { data: workflows, isLoading } = useWorkflows()
  const createWf = useCreateWorkflow()
  const deleteWf = useDeleteWorkflow()
  const loadFromDb = useWorkspaceStore((s) => s.loadFromDb)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const filtered = (workflows ?? []).filter(
    (w) => !search || w.name.includes(search),
  )

  const handleOpen = async (id: string) => {
    // Check if already open in a tab
    const tabs = useWorkspaceStore.getState().tabs
    const existing = tabs.find((t) => t.dbId === id)
    if (existing) {
      useWorkspaceStore.getState().setActiveTab(existing.id)
      return
    }
    setLoadingId(id)
    try {
      const resp = await fetch(`/api/v1/workflows/${id}`)
      if (!resp.ok) throw new Error('Failed to load workflow')
      const full = await resp.json()
      loadFromDb(full)
    } catch (err) {
      console.error('Failed to load workflow', err)
    } finally {
      setLoadingId(null)
    }
  }

  const handleCreate = () => {
    createWf.mutate({ name: `工作流 ${(workflows?.length ?? 0) + 1}` })
  }

  return (
    <FloatingPanel
      title="Workflows"
      searchPlaceholder="Search Workflows..."
      onSearch={setSearch}
      actions={
        <button
          onClick={handleCreate}
          disabled={createWf.isPending}
          style={{
            padding: '3px 10px',
            fontSize: 10,
            borderRadius: 4,
            border: '1px solid var(--accent)',
            background: 'var(--accent)',
            color: '#fff',
            cursor: createWf.isPending ? 'wait' : 'pointer',
            opacity: createWf.isPending ? 0.6 : 1,
          }}
        >
          + 新建
        </button>
      }
    >
      {isLoading && (
        <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>加载中...</div>
      )}
      {!isLoading && filtered.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--muted)', padding: 10 }}>暂无工作流</div>
      )}
      {filtered.map((w) => (
        <PresetCard key={w.id} onClick={() => handleOpen(w.id)} loading={loadingId === w.id}>
          <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
              {w.name}
            </div>
            <div className="flex items-center gap-1.5">
              <StatusBadge status={w.status} />
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm(`删除工作流「${w.name}」？`)) {
                    deleteWf.mutate(w.id)
                  }
                }}
                style={{
                  padding: '1px 5px',
                  fontSize: 9,
                  borderRadius: 3,
                  border: '1px solid var(--border)',
                  background: 'none',
                  color: 'var(--muted)',
                  cursor: 'pointer',
                }}
              >
                ×
              </button>
            </div>
          </div>
          <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, color: 'var(--muted)' }}>
            {w.is_template && <Tag>模板</Tag>}
          </div>
        </PresetCard>
      ))}
    </FloatingPanel>
  )
}

function StatusBadge({ status }: { status: string }) {
  const isPublished = status === 'published'
  return (
    <span
      style={{
        fontSize: 9,
        padding: '1px 5px',
        borderRadius: 3,
        background: isPublished ? 'rgba(52,199,89,0.15)' : 'rgba(255,255,255,0.06)',
        color: isPublished ? '#34c759' : 'var(--muted)',
        border: `1px solid ${isPublished ? 'rgba(52,199,89,0.3)' : 'var(--border)'}`,
      }}
    >
      {isPublished ? 'published' : 'draft'}
    </span>
  )
}

function PresetCard({ children, onClick, loading }: { children: React.ReactNode; onClick?: () => void; loading?: boolean }) {
  return (
    <div
      className="rounded-md cursor-pointer mb-1.5"
      onClick={onClick}
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        padding: 10,
        transition: 'all 0.12s',
        opacity: loading ? 0.6 : 1,
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
