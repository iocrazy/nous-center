import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import IconRail from './components/layout/IconRail'
import Topbar from './components/layout/Topbar'
import WorkflowTabs from './components/layout/WorkflowTabs'
import NodeEditor from './components/nodes/NodeEditor'
import ToastContainer from './components/common/ToastContainer'
import { usePanelStore, type OverlayId } from './stores/panel'

const ROUTE_TO_OVERLAY: Record<string, OverlayId> = {
  '/models': 'models',
  '/agents': 'agents',
  '/settings': 'settings',
  '/dashboard': 'dashboard',
  '/api': 'api-management',
}

/** Syncs the current URL to the panel store's activeOverlay */
function RouteSync() {
  const location = useLocation()
  const setOverlay = usePanelStore((s) => s.setOverlay)

  useEffect(() => {
    const overlay = ROUTE_TO_OVERLAY[location.pathname] ?? null
    usePanelStore.getState().activeOverlay !== overlay && setOverlay(overlay)
  }, [location.pathname, setOverlay])

  return null
}

function MainLayout() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
      <RouteSync />
      <IconRail />
      <div className="flex-1 flex flex-col overflow-hidden">
        <WorkflowTabs />
        <Topbar />
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
        <Route path="/models" element={<MainLayout />} />
        <Route path="/agents" element={<MainLayout />} />
        <Route path="/settings" element={<MainLayout />} />
        <Route path="/dashboard" element={<MainLayout />} />
        <Route path="/api" element={<MainLayout />} />
      </Routes>
    </BrowserRouter>
  )
}
