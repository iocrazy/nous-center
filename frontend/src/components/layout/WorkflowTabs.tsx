import { useState, useCallback, useEffect } from 'react'
import { X, Plus } from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useWorkspaceStore } from '../../stores/workspace'
import ContextMenu, { type MenuItem } from '../ui/ContextMenu'

interface TabMenuState {
  visible: boolean
  position: { x: number; y: number }
  tabId: string
}

export default function WorkflowTabs() {
  const { tabs, activeTabId, setActiveTab, addTab, removeTab, renameTab } = useWorkspaceStore()
  const qc = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()

  const [menu, setMenu] = useState<TabMenuState>({ visible: false, position: { x: 0, y: 0 }, tabId: '' })

  // Keep URL in sync with active tab's dbId:
  // - at mount / after full refresh on `/`, push URL to /workflows/:dbId
  // - when a freshly-created tab first auto-saves and gets its dbId, same
  // - when switching between saved tabs (handleTabClick also does this;
  //   this effect covers setActiveTab called from elsewhere)
  // Only touch the URL when we're on the workflow-editor surface (/, /workflows,
  // or /workflows/*) — don't stomp on /models, /agents, /settings, etc.
  const activeTab = tabs.find((t) => t.id === activeTabId)
  const activeDbId = activeTab?.dbId ?? null
  useEffect(() => {
    const p = location.pathname
    const onEditor = p === '/' || p === '/workflows' || p.startsWith('/workflows/')
    if (!onEditor) return
    const target = activeDbId ? `/workflows/${activeDbId}` : '/workflows'
    if (p !== target) navigate(target, { replace: true })
  }, [activeDbId, location.pathname, navigate])

  const handleTabClick = (id: string) => {
    setActiveTab(id)
    const tab = tabs.find((t) => t.id === id)
    const target = tab?.dbId ? `/workflows/${tab.dbId}` : '/workflows'
    if (location.pathname !== target) navigate(target)
  }

  const closeMenu = useCallback(() => setMenu((m) => ({ ...m, visible: false })), [])

  const handleContextMenu = useCallback((e: React.MouseEvent, tabId: string) => {
    e.preventDefault()
    setMenu({ visible: true, position: { x: e.clientX, y: e.clientY }, tabId })
  }, [])

  const menuTab = tabs.find((t) => t.id === menu.tabId)
  const menuTabIndex = tabs.findIndex((t) => t.id === menu.tabId)

  const menuItems: MenuItem[] = menuTab
    ? [
        {
          label: '重命名',
          onClick: () => {
            const name = window.prompt('输入新名称', menuTab.name)
            if (name && name.trim()) {
              renameTab(menuTab.id, name.trim())
              // Refresh sidebar workflow list after backend sync
              setTimeout(() => qc.invalidateQueries({ queryKey: ['workflows'] }), 500)
            }
          },
        },
        {
          label: '复制',
          onClick: () => {
            // Duplicate the workflow as a new tab
            const wf = tabs.find((t) => t.id === menuTab.id)?.workflow
            if (!wf) return
            const newTab = {
              id: '',  // will be set by addTab
              name: `${menuTab.name} (副本)`,
              nodes: JSON.parse(JSON.stringify(wf.nodes)),
              edges: JSON.parse(JSON.stringify(wf.edges)),
            }
            addTab(newTab.name)
            // After addTab, the new tab is active. Set its workflow content.
            const store = useWorkspaceStore.getState()
            const activeTab = store.tabs.find((t) => t.id === store.activeTabId)
            if (activeTab) {
              useWorkspaceStore.setState({
                tabs: store.tabs.map((t) =>
                  t.id === activeTab.id
                    ? { ...t, workflow: { ...t.workflow, nodes: newTab.nodes, edges: newTab.edges } }
                    : t
                ),
              })
            }
          },
        },
        { label: '', divider: true },
        {
          label: '关闭标签',
          onClick: () => removeTab(menuTab.id),
          disabled: tabs.length <= 1,
        },
        {
          label: '关闭左侧标签',
          onClick: () => {
            const leftIds = tabs.slice(0, menuTabIndex).map((t) => t.id)
            leftIds.forEach((id) => useWorkspaceStore.getState().removeTab(id))
          },
          disabled: menuTabIndex === 0,
        },
        {
          label: '关闭右侧标签',
          onClick: () => {
            const rightIds = tabs.slice(menuTabIndex + 1).map((t) => t.id)
            rightIds.forEach((id) => useWorkspaceStore.getState().removeTab(id))
          },
          disabled: menuTabIndex === tabs.length - 1,
        },
        {
          label: '关闭其他标签',
          onClick: () => {
            const otherIds = tabs.filter((t) => t.id !== menuTab.id).map((t) => t.id)
            otherIds.forEach((id) => useWorkspaceStore.getState().removeTab(id))
            setActiveTab(menuTab.id)
          },
          disabled: tabs.length <= 1,
        },
      ]
    : []

  return (
    <>
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
              onContextMenu={(e) => handleContextMenu(e, tab.id)}
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
                <span
                  role="button"
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
                </span>
              )}
            </button>
          </div>
        ))}
        <button
          onClick={() => {
            addTab()
            // New tab has no dbId yet — URL goes to /workflows until first save.
            if (location.pathname !== '/workflows') navigate('/workflows')
          }}
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

      {menu.visible && (
        <ContextMenu items={menuItems} position={menu.position} onClose={closeMenu} />
      )}
    </>
  )
}
