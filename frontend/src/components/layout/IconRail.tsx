import {
  LayoutDashboard,
  Layers,
  CircuitBoard,
  GitBranch,
  Settings,
  SlidersHorizontal,
  Link,
  Bot,
  Sun,
  Moon,
  Monitor,
  ListTodo,
} from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
import { usePanelStore, type PanelId, type OverlayId } from '../../stores/panel'
import { useThemeStore } from '../../stores/theme'
import { useExecutionStore } from '../../stores/execution'
import { useTasks } from '../../api/tasks'

const OVERLAY_ROUTES: Record<OverlayId, string> = {
  dashboard: '/dashboard',
  models: '/models',
  'api-management': '/api',
  agents: '/agents',
  settings: '/settings',
  'preset-detail': '/', // no dedicated route
}

const PANEL_ITEMS: { id: PanelId; icon: typeof CircuitBoard; label: string }[] = [
  { id: 'nodes', icon: CircuitBoard, label: 'Nodes' },
  { id: 'workflows', icon: GitBranch, label: 'Workflows' },
  { id: 'presets', icon: Settings, label: 'Presets' },
]

const OVERLAY_ITEMS: { id: OverlayId; icon: typeof LayoutDashboard; label: string }[] = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'models', icon: Layers, label: 'Models' },
  { id: 'api-management', icon: Link, label: 'API Management' },
  { id: 'agents', icon: Bot, label: 'Agents' },
]

export default function IconRail() {
  const { activePanel, activeOverlay, togglePanel } = usePanelStore()
  const { mode, setMode } = useThemeStore()
  const navigate = useNavigate()
  const location = useLocation()

  const navigateOverlay = (id: OverlayId) => {
    const target = OVERLAY_ROUTES[id]
    // Toggle: if already on that route, go home
    if (activeOverlay === id || location.pathname === target) {
      navigate('/')
    } else {
      navigate(target)
    }
  }

  return (
    <div
      className="flex flex-col items-center shrink-0 z-20"
      style={{
        width: 48,
        background: 'var(--bg-accent)',
        borderRight: '1px solid var(--border)',
        padding: '8px 0',
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center justify-center font-extrabold text-white text-[13px] mb-3"
        style={{
          width: 28,
          height: 28,
          borderRadius: 6,
          background: 'var(--accent)',
        }}
      >
        N
      </div>

      {/* Overlay buttons (Dashboard, Models) */}
      {OVERLAY_ITEMS.map(({ id, icon: Icon, label }) => (
        <RailButton
          key={id}
          active={activeOverlay === id}
          onClick={() => navigateOverlay(id)}
          label={label}
        >
          <Icon size={18} />
        </RailButton>
      ))}

      {/* Separator */}
      <div
        className="my-1.5"
        style={{ width: 24, height: 1, background: 'var(--border)' }}
      />

      {/* Panel buttons */}
      {PANEL_ITEMS.map(({ id, icon: Icon, label }) => (
        <RailButton
          key={id}
          active={activePanel === id && !activeOverlay}
          onClick={() => {
            if (location.pathname !== '/') navigate('/')
            togglePanel(id)
          }}
          label={label}
        >
          <Icon size={18} />
        </RailButton>
      ))}

      {/* Bottom section */}
      <div className="mt-auto flex flex-col items-center gap-1">
        <TaskRailButton />

        <RailButton
          active={activeOverlay === 'settings'}
          onClick={() => navigateOverlay('settings')}
          label="Settings"
        >
          <SlidersHorizontal size={18} />
        </RailButton>

        <div className="my-1" style={{ width: 24, height: 1, background: 'var(--border)' }} />

        <ThemeButton active={mode === 'light'} onClick={() => setMode('light')}>
          <Sun size={12} />
        </ThemeButton>
        <ThemeButton active={mode === 'dark'} onClick={() => setMode('dark')}>
          <Moon size={12} />
        </ThemeButton>
        <ThemeButton active={mode === 'auto'} onClick={() => setMode('auto')}>
          <Monitor size={12} />
        </ThemeButton>
        <div className="flex items-center gap-1 mt-1" style={{ fontSize: 8, color: 'var(--muted)' }}>
          <span
            className="inline-block rounded-full"
            style={{ width: 5, height: 5, background: 'var(--ok)' }}
          />
          GPU
        </div>
      </div>
    </div>
  )
}

function RailButton({
  active,
  onClick,
  label,
  children,
}: {
  active: boolean
  onClick: () => void
  label: string
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className="group relative flex items-center justify-center mb-0.5"
      style={{
        width: 36,
        height: 36,
        borderRadius: 6,
        border: 'none',
        background: active ? 'var(--accent-subtle)' : 'transparent',
        color: active ? 'var(--accent)' : 'var(--muted)',
        cursor: 'pointer',
        transition: 'all 0.12s',
      }}
      onMouseEnter={(e) => {
        if (!active) {
          e.currentTarget.style.background = 'var(--bg-hover)'
          e.currentTarget.style.color = 'var(--text)'
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          e.currentTarget.style.background = 'transparent'
          e.currentTarget.style.color = 'var(--muted)'
        }
      }}
    >
      {children}
      <span
        className="hidden group-hover:block absolute pointer-events-none"
        style={{
          left: 44,
          top: '50%',
          transform: 'translateY(-50%)',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          padding: '3px 8px',
          fontSize: 10,
          whiteSpace: 'nowrap',
          color: 'var(--text)',
          zIndex: 100,
        }}
      >
        {label}
      </span>
    </button>
  )
}

function TaskRailButton() {
  const { taskPanelOpen, toggleTaskPanel } = useExecutionStore()
  const { data: tasks } = useTasks()
  const runningCount = tasks?.filter((t) => t.status === 'running').length ?? 0

  return (
    <div className="relative">
      <RailButton active={taskPanelOpen} onClick={toggleTaskPanel} label="Tasks">
        <ListTodo size={18} />
      </RailButton>
      {runningCount > 0 && (
        <span
          className="absolute pointer-events-none"
          style={{
            top: 2,
            right: 2,
            minWidth: 14,
            height: 14,
            borderRadius: 7,
            background: 'var(--accent)',
            color: '#fff',
            fontSize: 9,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '0 3px',
          }}
        >
          {runningCount}
        </span>
      )}
    </div>
  )
}

function ThemeButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center justify-center"
      style={{
        width: 32,
        height: 28,
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: active ? 'var(--bg-elevated)' : 'transparent',
        color: active ? 'var(--text-strong)' : 'var(--muted)',
        cursor: 'pointer',
        fontSize: 11,
      }}
    >
      {children}
    </button>
  )
}
