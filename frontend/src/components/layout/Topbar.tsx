import { useEffect, useCallback } from 'react'
import { ArrowLeft } from 'lucide-react'
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'
import { useExecutionStore } from '../../stores/execution'
import { executeWorkflow } from '../../utils/workflowExecutor'
import { useToastStore } from '../../stores/toast'

export default function Topbar() {
  const { tabs, activeTabId } = useWorkspaceStore()
  const getActiveWorkflow = useWorkspaceStore((s) => s.getActiveWorkflow)
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const { activeOverlay, setOverlay } = usePanelStore()
  const { isRunning, start, succeed, fail } = useExecutionStore()
  const toast = useToastStore((s) => s.add)

  const activeTab = tabs.find((t) => t.id === activeTabId)

  const overlayTitle = activeOverlay === 'dashboard' ? 'Dashboard' : activeOverlay === 'models' ? 'Models' : activeOverlay === 'settings' ? '设置' : activeOverlay === 'preset-detail' ? '预设详情' : activeOverlay === 'api-management' ? 'API 管理' : null

  const handleRun = async () => {
    if (isRunning) return
    const workflow = getActiveWorkflow()
    start()

    try {
      const result = await executeWorkflow(workflow)
      succeed(result)
      toast('生成完成', 'success')

      // Update output node with result
      const outputNode = workflow.nodes.find((n) => n.type === 'output')
      if (outputNode) {
        updateNode(outputNode.id, {
          audioBase64: result.audioBase64,
          sampleRate: result.sampleRate,
          duration: result.duration,
        })
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      fail(msg)
      toast(msg, 'error')
    }
  }

  // Ctrl+Enter shortcut
  const handleRunCb = useCallback(() => { handleRun() }, [isRunning])
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        handleRunCb()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [handleRunCb])

  return (
    <div
      className="flex items-center gap-2 shrink-0 z-[18]"
      style={{
        height: 36,
        background: 'rgba(18,20,26,0.9)',
        borderBottom: '1px solid var(--border)',
        padding: '0 12px',
        backdropFilter: 'blur(8px)',
      }}
    >
      {activeOverlay && (
        <button
          onClick={() => setOverlay(null)}
          className="flex items-center justify-center"
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
          <ArrowLeft size={12} />
        </button>
      )}

      <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-strong)' }}>
        {overlayTitle ?? 'Workspace'}
      </span>

      {!activeOverlay && activeTab && (
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          · {activeTab.name}
        </span>
      )}

      <div className="ml-auto flex gap-1.5">
        {!activeOverlay && (
          <>
            <TopbarButton>Templates</TopbarButton>
            <TopbarButton>Clear</TopbarButton>
            <TopbarButton primary onClick={handleRun} disabled={isRunning}>
              {isRunning ? '⏳ Running...' : '▶ Run'}
            </TopbarButton>
          </>
        )}
      </div>
    </div>
  )
}

function TopbarButton({
  primary,
  children,
  onClick,
  disabled,
}: {
  primary?: boolean
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: '4px 10px',
        fontSize: 10,
        borderRadius: 4,
        border: `1px solid ${primary ? 'var(--accent)' : 'var(--border)'}`,
        background: primary ? 'var(--accent)' : 'none',
        color: primary ? '#fff' : 'var(--muted)',
        cursor: disabled ? 'wait' : 'pointer',
        opacity: disabled ? 0.6 : 1,
      }}
    >
      {children}
    </button>
  )
}
