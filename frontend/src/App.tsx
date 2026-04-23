import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation, useParams } from 'react-router-dom'
import IconRail from './components/layout/IconRail'
import Topbar from './components/layout/Topbar'
import WorkflowTabs from './components/layout/WorkflowTabs'
import NodeEditor from './components/nodes/NodeEditor'
import ToastContainer from './components/common/ToastContainer'
import { usePanelStore, type OverlayId } from './stores/panel'
import { useWorkspaceStore } from './stores/workspace'
import { apiFetch } from './api/client'
import type { WorkflowFull } from './api/workflows'
import { useToastStore } from './stores/toast'

const ROUTE_TO_OVERLAY: Record<string, OverlayId> = {
  '/models': 'models',
  '/services': 'services',
  '/apps': 'apps',
  '/agents': 'agents',
  '/settings': 'settings',
  '/dashboard': 'dashboard',
  '/api-management': 'api-management',
  '/logs': 'logs',
  '/node-packages': 'node-packages',
  '/usage': 'usage',
}

/** Syncs the current URL to the panel store's activeOverlay */
function RouteSync() {
  const location = useLocation()
  const setOverlay = usePanelStore((s) => s.setOverlay)

  useEffect(() => {
    // `/workflows` and `/workflows/:id` share the workflow editor (no overlay).
    if (location.pathname.startsWith('/workflows')) {
      if (usePanelStore.getState().activeOverlay !== null) setOverlay(null)
      return
    }
    // `/services/:id` lights up the same rail slot as the list and routes
    // through the dedicated `service-detail` overlay so NodeEditor knows
    // which view to mount.
    if (location.pathname.startsWith('/services/')) {
      if (usePanelStore.getState().activeOverlay !== 'service-detail') {
        setOverlay('service-detail')
      }
      return
    }
    const overlay = ROUTE_TO_OVERLAY[location.pathname] ?? null
    if (usePanelStore.getState().activeOverlay !== overlay) setOverlay(overlay)
  }, [location.pathname, setOverlay])

  return null
}

/** When URL is /workflows/:id, activate (or fetch + load) that workflow. */
function WorkflowRouteLoader() {
  const { id } = useParams<{ id: string }>()
  const activateByDbId = useWorkspaceStore((s) => s.activateByDbId)
  const loadFromDb = useWorkspaceStore((s) => s.loadFromDb)

  useEffect(() => {
    if (!id) return
    if (activateByDbId(id)) return
    // Not yet in tabs — fetch from backend and open as a new tab.
    apiFetch<WorkflowFull>(`/api/v1/workflows/${encodeURIComponent(id)}`)
      .then(loadFromDb)
      .catch((err) => {
        useToastStore.getState().add(`加载工作流失败: ${err.message ?? err}`, 'error')
      })
  }, [id, activateByDbId, loadFromDb])

  return null
}

function MainLayout({ workflowRoute }: { workflowRoute?: boolean }) {
  const activeOverlay = usePanelStore((s) => s.activeOverlay)
  const isWorkflowView = !activeOverlay

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
      <RouteSync />
      {workflowRoute && <WorkflowRouteLoader />}
      <IconRail />
      <div className="flex-1 flex flex-col overflow-hidden">
        {isWorkflowView && <WorkflowTabs />}
        {isWorkflowView && <Topbar />}
        <NodeEditor />
      </div>
      <ToastContainer />
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />} />
        <Route path="/workflows" element={<MainLayout />} />
        <Route path="/workflows/:id" element={<MainLayout workflowRoute />} />
        <Route path="/models" element={<MainLayout />} />
        <Route path="/services" element={<MainLayout />} />
        <Route path="/apps" element={<MainLayout />} />
        <Route path="/agents" element={<MainLayout />} />
        <Route path="/settings" element={<MainLayout />} />
        <Route path="/dashboard" element={<MainLayout />} />
        <Route path="/api-management" element={<MainLayout />} />
        <Route path="/logs" element={<MainLayout />} />
        <Route path="/node-packages" element={<MainLayout />} />
        <Route path="/usage" element={<MainLayout />} />
        <Route path="/services/:id" element={<MainLayout />} />
      </Routes>
    </BrowserRouter>
  )
}
