import { X, Plus } from 'lucide-react'
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'

export default function WorkflowTabs() {
  const { tabs, activeTabId, setActiveTab, addTab, removeTab } = useWorkspaceStore()
  const { setOverlay } = usePanelStore()

  const handleTabClick = (id: string) => {
    setActiveTab(id)
    setOverlay(null)
  }

  return (
    <div
      className="flex items-stretch shrink-0 z-[18] overflow-x-auto"
      style={{
        height: 32,
        background: 'var(--bg-accent)',
        borderTop: '1px solid var(--border)',
        padding: '0 4px',
        gap: 0,
      }}
    >
      {tabs.map((tab, i) => (
        <div key={tab.id} className="flex items-stretch">
          {i > 0 && (
            <div
              className="shrink-0 self-center"
              style={{ width: 1, height: 16, background: 'var(--border)', margin: '0 0' }}
            />
          )}
          <button
            onClick={() => handleTabClick(tab.id)}
            className="group flex items-center gap-1.5 whitespace-nowrap relative"
            style={{
              padding: '0 12px',
              fontSize: 11,
              border: 'none',
              background: 'none',
              color: tab.id === activeTabId ? 'var(--text-strong)' : 'var(--muted)',
              cursor: 'pointer',
              borderBottom: tab.id === activeTabId ? '2px solid var(--accent)' : '2px solid transparent',
              transition: 'all 0.1s',
            }}
          >
            <span>{tab.name}</span>
            {tab.isDirty && (
              <span style={{ color: 'var(--accent)', fontSize: 8 }}>●</span>
            )}
            {tabs.length > 1 && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeTab(tab.id)
                }}
                className="invisible group-hover:visible flex items-center justify-center"
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  border: 'none',
                  background: 'none',
                  color: 'var(--muted-strong)',
                  cursor: 'pointer',
                  fontSize: 9,
                }}
              >
                <X size={9} />
              </button>
            )}
          </button>
        </div>
      ))}
      <button
        onClick={() => addTab()}
        className="flex items-center justify-center shrink-0"
        style={{
          width: 28,
          border: 'none',
          background: 'none',
          color: 'var(--muted-strong)',
          cursor: 'pointer',
          fontSize: 14,
        }}
      >
        <Plus size={14} />
      </button>
    </div>
  )
}
