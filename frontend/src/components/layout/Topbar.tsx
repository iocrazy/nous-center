import { useEffect, useCallback, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import PublishDialog from '../workflow/PublishDialog'
import { useNavigate, useLocation } from 'react-router-dom'
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'
import { useExecutionStore } from '../../stores/execution'
import { executeWorkflow } from '../../utils/workflowExecutor'
import { useToastStore } from '../../stores/toast'
import { useUnpublishWorkflow } from '../../api/workflows'
import { useNotificationStore } from '../../stores/notifications'

export default function Topbar() {
  const { tabs, activeTabId } = useWorkspaceStore()
  const getActiveWorkflow = useWorkspaceStore((s) => s.getActiveWorkflow)
  const setWorkflow = useWorkspaceStore((s) => s.setWorkflow)
  const { activeOverlay } = usePanelStore()
  const navigate = useNavigate()
  const { isRunning, progress, currentNodeType, start, succeed, fail, resetNodeStates, bumpTaskBadge } = useExecutionStore()
  const toast = useToastStore((s) => s.add)
  const unpublishWf = useUnpublishWorkflow()
  const requestNotifyPermission = useNotificationStore((s) => s.requestPermission)

  const activeTab = tabs.find((t) => t.id === activeTabId)
  const isPublished = activeTab?.workflow?.status === 'published'
  const [showPublishWizard, setShowPublishWizard] = useState(false)
  const location = useLocation()

  // m08 列表卡"发布为服务"按钮 → 跳 /workflows/:id?publish=1
  // → 这里检测到 query 自动弹发布对话框 + 清掉 query。要等 activeTab
  // 加载完才弹（不然 PublishDialog 拿不到 nodes 数据）。
  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get('publish') !== '1') return
    if (!activeTab?.workflow) return
    setShowPublishWizard(true)
    // replace 把 query 清掉，避免刷新或返回又弹
    navigate(location.pathname, { replace: true })
  }, [activeTab, location.search, location.pathname, navigate])

  const overlayTitle = activeOverlay === 'dashboard' ? 'Dashboard' : activeOverlay === 'models' ? 'Models' : activeOverlay === 'settings' ? '设置' : activeOverlay === 'preset-detail' ? '预设详情' : activeOverlay === 'api-keys-list' ? 'API Key' : activeOverlay === 'api-key-detail' ? 'API Key 详情' : null

  const handleRun = async () => {
    if (isRunning) return
    // spec §6.6：浏览器通知权限「首次询问」—— 在用户首次点 Run 时问
    // （比页面加载时弹更得体）。requestPermission 内部幂等：已问过就直接读快照。
    void requestNotifyPermission()
    const workflow = getActiveWorkflow()
    start()

    try {
      // Lane S（D17）：executeWorkflow 入队后立即返回 { task_id }，不再阻塞到完成。
      // 反馈 UX（spec §6.3 DD4）：toast「已入队」+ IconRail badge 计数 +
      // 面板【不】自动打开（用户点 IconRail 任务图标进面板）。
      const result = await executeWorkflow(workflow)
      const taskId = (result as { task_id?: string })?.task_id
      bumpTaskBadge()
      toast(taskId ? `任务已入队 · ${taskId}` : '任务已入队', 'info')
      // 入队即结束「本次 Run」的 UI busy 态 —— 后续进度由 TaskPanel 泳道接管。
      succeed(null)
      resetNodeStates()
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      fail(msg)
      toast(msg, 'error')
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
            {activeTab?.workflow && (
              isPublished ? (
                <button
                  onClick={() => activeTab?.dbId && unpublishWf.mutate(activeTab.dbId)}
                  disabled={unpublishWf.isPending || !activeTab?.dbId}
                  title="取消发布后服务将下线，新调用会 404"
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: 'none',
                    background: 'var(--ok, #34c759)',
                    color: '#fff',
                    cursor: unpublishWf.isPending ? 'wait' : 'pointer',
                    fontSize: 13,
                    opacity: unpublishWf.isPending ? 0.6 : 1,
                  }}
                >
                  {unpublishWf.isPending ? '取消中…' : '取消发布'}
                </button>
              ) : (
                <button
                  onClick={() => setShowPublishWizard(true)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: 'none',
                    background: 'var(--accent)',
                    color: '#fff',
                    cursor: 'pointer',
                    fontSize: 13,
                  }}
                >
                  发布服务
                </button>
              )
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
