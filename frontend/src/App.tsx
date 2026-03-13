import IconRail from './components/layout/IconRail'
import Topbar from './components/layout/Topbar'
import WorkflowTabs from './components/layout/WorkflowTabs'
import NodeEditor from './components/nodes/NodeEditor'
import ToastContainer from './components/common/ToastContainer'

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg)' }}>
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
