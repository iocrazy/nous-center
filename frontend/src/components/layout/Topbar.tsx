import { useEffect, useCallback, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import PublishDialog from '../workflow/PublishDialog'
import { useNavigate } from 'react-router-dom'
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'
import { useExecutionStore } from '../../stores/execution'
import { executeWorkflow } from '../../utils/workflowExecutor'
import { useToastStore } from '../../stores/toast'
import { usePublishWorkflow, useUnpublishWorkflow } from '../../api/workflows'

export default function Topbar() {
  const { tabs, activeTabId } = useWorkspaceStore()
  const getActiveWorkflow = useWorkspaceStore((s) => s.getActiveWorkflow)
  const setWorkflow = useWorkspaceStore((s) => s.setWorkflow)
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const { activeOverlay } = usePanelStore()
  const navigate = useNavigate()
  const { isRunning, progress, currentNodeType, start, succeed, fail, resetNodeStates } = useExecutionStore()
  const toast = useToastStore((s) => s.add)
  const publishWf = usePublishWorkflow()
  const unpublishWf = useUnpublishWorkflow()

  const activeTab = tabs.find((t) => t.id === activeTabId)
  const isPublished = activeTab?.workflow?.status === 'published'
  const [showPublishWizard, setShowPublishWizard] = useState(false)

  const overlayTitle = activeOverlay === 'dashboard' ? 'Dashboard' : activeOverlay === 'models' ? 'Models' : activeOverlay === 'settings' ? '设置' : activeOverlay === 'preset-detail' ? '预设详情' : activeOverlay === 'api-keys-list' ? 'API Key' : activeOverlay === 'api-key-detail' ? 'API Key 详情' : null

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

      // Keep final states visible, then reset
      setTimeout(() => resetNodeStates(), 3000)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      fail(msg)
      toast(msg, 'error')

      // Keep error states visible, then reset
      setTimeout(() => resetNodeStates(), 5000)
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
        background: 'var(--bg-accent)',
        borderBottom: '1px solid var(--border)',
        padding: '0 12px',
        backdropFilter: 'blur(8px)',
        position: 'relative',
      }}
    >
      {activeOverlay && (
        <button
          onClick={() => navigate('/')}
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
        <>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>
            · {activeTab.name}
          </span>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: activeTab.isDirty
                ? '#ff9f0a'
                : activeTab.dbId
                  ? '#34c759'
                  : 'transparent',
              display: 'inline-block',
              marginLeft: 2,
            }}
            title={activeTab.isDirty ? '未保存' : activeTab.dbId ? '已保存' : ''}
          />
          {activeTab.dbId && (
            <span
              style={{
                fontSize: 9,
                padding: '1px 6px',
                borderRadius: 3,
                background: isPublished ? 'rgba(52,199,89,0.15)' : 'rgba(255,255,255,0.06)',
                color: isPublished ? '#34c759' : 'var(--muted)',
                border: `1px solid ${isPublished ? 'rgba(52,199,89,0.3)' : 'var(--border)'}`,
              }}
            >
              {isPublished ? 'published' : 'draft'}
            </span>
          )}
        </>
      )}

      {/* Current node execution indicator */}
      {isRunning && currentNodeType && (
        <span style={{ fontSize: 10, color: 'var(--ok)', marginLeft: 8, flexShrink: 0 }}>
          ● {progress}% — {currentNodeType}
        </span>
      )}

      <div className="ml-auto flex gap-1.5">
        {!activeOverlay && (
          <>
            <TopbarButton>Templates</TopbarButton>
            <TopbarButton onClick={() => {
              if (window.confirm('清空当前工作流的所有节点和连线？')) {
                const wf = getActiveWorkflow()
                setWorkflow({ ...wf, nodes: [], edges: [] })
              }
            }}>Clear</TopbarButton>
            <TopbarButton primary onClick={handleRun} disabled={isRunning}>
              {isRunning ? '⏳ Running...' : '▶ Run'}
            </TopbarButton>
            {activeTab?.dbId && (
              <button
                onClick={() => {
                  if (isPublished) {
                    unpublishWf.mutate(activeTab.dbId!)
                  } else {
                    publishWf.mutate(activeTab.dbId!)
                  }
                }}
                disabled={publishWf.isPending || unpublishWf.isPending}
                style={{
                  padding: '4px 12px',
                  fontSize: 11,
                  borderRadius: 4,
                  border: '1px solid var(--border)',
                  background: isPublished ? 'none' : 'var(--ok)',
                  color: isPublished ? 'var(--muted)' : '#fff',
                  cursor: publishWf.isPending || unpublishWf.isPending ? 'wait' : 'pointer',
                  opacity: publishWf.isPending || unpublishWf.isPending ? 0.6 : 1,
                }}
              >
                {isPublished ? '下线' : '发布'}
              </button>
            )}
            {!isPublished && (
              <button
                onClick={() => setShowPublishWizard(true)}
                style={{ padding: '6px 14px', borderRadius: 6, border: 'none', background: 'var(--accent)', color: '#fff', cursor: 'pointer', fontSize: 13 }}
              >
                发布服务
              </button>
            )}
            {activeTab?.workflow && (
              <PublishDialog
                open={showPublishWizard}
                onClose={() => setShowPublishWizard(false)}
                workflowId={String(activeTab.workflow.id)}
                nodes={(activeTab.workflow.nodes ?? []) as Array<{ id: string; type?: string; data?: Record<string, unknown> }>}
              />
            )}
          </>
        )}
      </div>

      {/* Bottom progress bar */}
      {isRunning && (
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 3,
          background: 'var(--border)',
        }}>
          <div style={{
            height: '100%',
            width: `${progress}%`,
            background: 'var(--ok)',
            transition: 'width 0.3s ease',
            borderRadius: '0 1px 1px 0',
          }} />
        </div>
      )}
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
