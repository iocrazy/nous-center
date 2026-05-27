/**
 * GlobalTopbar — 全局顶部 nav(PR-2,任务面板重置)。
 *
 * 复刻 spec mockup `docs/superpowers/specs/assets/2026-05-27-task-panel-reset/variant-final.html`
 * 的 .topbar 结构。布局:
 *
 *   nous-logo  · Workflow · Image · TTS · LLM · Models                    infra healthy · 3 GPU  ⌕  ⋮
 *   ─────────────────────── 5 服务 tab(主路由)───────────────────  ── 全局状态 / 搜索 / admin 下拉
 *
 * IconRail 在 PR-2 删除;它的 6 个 admin 项(Dashboard/服务/API Key/用量/日志/设置)+ 主题切换 + 退出
 * 全部塞进右侧 admin dropdown(adminMenuOpen state)。
 */
import { useState, useRef, useEffect } from 'react'
import {
  Search, MoreVertical,
  LayoutDashboard, Activity, KeyRound, BarChart3, ScrollText, SlidersHorizontal,
  Sun, Moon, Monitor, LogOut, GitBranch, Image as ImageIcon, Mic, MessageSquare, Database,
} from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useThemeStore } from '../../stores/theme'
import { useAdminLogout, useAdminMe } from '../../api/admin'
import type { LucideIcon } from 'lucide-react'

type ServiceTabId = 'workflow' | 'image' | 'tts' | 'llm' | 'models'

const SERVICE_TABS: { id: ServiceTabId; label: string; icon: LucideIcon; route: string }[] = [
  { id: 'workflow', label: 'Workflow', icon: GitBranch,      route: '/workflows' },
  { id: 'image',    label: 'Image',    icon: ImageIcon,      route: '/image' },
  { id: 'tts',      label: 'TTS',      icon: Mic,            route: '/tts' },
  { id: 'llm',      label: 'LLM',      icon: MessageSquare,  route: '/llm' },
  { id: 'models',   label: 'Models',   icon: Database,       route: '/models' },
]

type AdminMenuItem = { id: string; label: string; icon: LucideIcon; route: string }

const ADMIN_ITEMS: AdminMenuItem[] = [
  { id: 'dashboard',  label: 'Dashboard', icon: LayoutDashboard,     route: '/dashboard' },
  { id: 'services',   label: '服务',      icon: Activity,            route: '/services' },
  { id: 'api-keys',   label: 'API Key',   icon: KeyRound,            route: '/api-keys' },
  { id: 'usage',      label: '用量',      icon: BarChart3,           route: '/usage' },
  { id: 'logs',       label: '日志',      icon: ScrollText,          route: '/logs' },
  { id: 'settings',   label: '设置',      icon: SlidersHorizontal,   route: '/settings' },
]

function routeToTab(pathname: string): ServiceTabId {
  // /workflows / /workflows/:id / / (root) → workflow
  if (pathname === '/' || pathname.startsWith('/workflow')) return 'workflow'
  if (pathname.startsWith('/image')) return 'image'
  if (pathname.startsWith('/tts')) return 'tts'
  if (pathname.startsWith('/llm')) return 'llm'
  if (pathname.startsWith('/models')) return 'models'
  // admin overlay 路由 — 沿用 workflow tab 高亮(避免「我在设置页但没 tab 高亮」)
  return 'workflow'
}

export default function GlobalTopbar() {
  const navigate = useNavigate()
  const location = useLocation()
  const { mode, setMode } = useThemeStore()
  const me = useAdminMe()
  const logout = useAdminLogout()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  const activeTab = routeToTab(location.pathname)
  const canLogout = me.data?.login_required && me.data?.authenticated

  // 点外部关菜单
  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [menuOpen])

  const handleTabClick = (route: string) => {
    if (location.pathname !== route) navigate(route)
  }

  const handleAdminClick = (route: string) => {
    navigate(route)
    setMenuOpen(false)
  }

  const handleLogout = () => {
    logout.mutate(undefined, {
      onSuccess: () => {
        setMenuOpen(false)
        navigate('/')
      },
    })
  }

  return (
    <div
      className="flex items-center px-4 shrink-0 z-30"
      style={{
        height: 44,
        background: 'var(--tp-bg-panel)',
        borderBottom: '1px solid var(--tp-border-faint)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 font-bold text-[15px]" style={{ color: 'var(--tp-text)' }}>
        <span
          style={{
            width: 8, height: 8, borderRadius: 2,
            background: 'linear-gradient(135deg, var(--type-image), var(--type-tts) 50%, var(--type-llm))',
          }}
        />
        nous
      </div>

      {/* 5 服务 tab */}
      <nav className="flex gap-1 ml-8" role="navigation" aria-label="主导航">
        {SERVICE_TABS.map(({ id, label, icon: Icon, route }) => {
          const active = activeTab === id
          return (
            <button
              key={id}
              onClick={() => handleTabClick(route)}
              className="inline-flex items-center gap-1.5 px-3 h-7 text-xs rounded-md transition-colors"
              style={{
                background: active ? 'var(--tp-bg-card)' : 'transparent',
                color: active ? 'var(--tp-text)' : 'var(--tp-text-muted)',
                cursor: 'pointer',
              }}
              aria-current={active ? 'page' : undefined}
              data-tab-id={id}
            >
              <Icon size={14} />
              {label}
            </button>
          )
        })}
      </nav>

      <div className="flex-1" />

      {/* 全局状态 */}
      <div className="text-xs flex items-center gap-1.5 mr-4" style={{ color: 'var(--tp-text-muted)' }}>
        <span
          style={{
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--status-running)',
            boxShadow: '0 0 8px var(--status-running)',
          }}
        />
        infra <span style={{ color: 'var(--tp-text)', fontWeight: 500 }}>healthy</span>
      </div>

      {/* 搜索 placeholder(PR-2 仅 UI 占位;搜索功能后续 PR)*/}
      <button
        className="p-1.5 transition-colors"
        style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
        aria-label="搜索"
        title="搜索(占位)"
      >
        <Search size={16} />
      </button>

      {/* admin dropdown */}
      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setMenuOpen((o) => !o)}
          className="p-1.5 ml-1 transition-colors"
          style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
          aria-label="管理菜单"
          aria-expanded={menuOpen}
          aria-haspopup="menu"
        >
          <MoreVertical size={16} />
        </button>
        {menuOpen && (
          <div
            role="menu"
            className="absolute right-0 top-full mt-2 py-1 rounded-md"
            style={{
              minWidth: 200,
              background: 'var(--tp-bg-card)',
              border: '1px solid var(--tp-border-strong)',
              boxShadow: 'var(--shadow-card, 0 6px 16px rgba(0,0,0,0.3))',
            }}
          >
            {ADMIN_ITEMS.map(({ id, label, icon: Icon, route }) => (
              <button
                key={id}
                onClick={() => handleAdminClick(route)}
                role="menuitem"
                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors hover:bg-[var(--tp-bg-hover)]"
                style={{ color: 'var(--tp-text)', cursor: 'pointer' }}
              >
                <Icon size={14} style={{ color: 'var(--tp-text-muted)' }} />
                {label}
              </button>
            ))}
            <div className="my-1" style={{ height: 1, background: 'var(--tp-border-faint)' }} />
            {/* 主题切换 */}
            <div className="flex items-center gap-1 px-3 py-2">
              <ThemeBtn active={mode === 'light'} onClick={() => setMode('light')} aria="浅色主题"><Sun size={12} /></ThemeBtn>
              <ThemeBtn active={mode === 'dark'} onClick={() => setMode('dark')} aria="深色主题"><Moon size={12} /></ThemeBtn>
              <ThemeBtn active={mode === 'auto'} onClick={() => setMode('auto')} aria="跟随系统"><Monitor size={12} /></ThemeBtn>
            </div>
            {canLogout && (
              <>
                <div className="my-1" style={{ height: 1, background: 'var(--tp-border-faint)' }} />
                <button
                  onClick={handleLogout}
                  role="menuitem"
                  className="w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors hover:bg-[var(--tp-bg-hover)]"
                  style={{ color: 'var(--tp-text)', cursor: 'pointer' }}
                >
                  <LogOut size={14} style={{ color: 'var(--tp-text-muted)' }} />
                  退出登录
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ThemeBtn({
  active, onClick, children, aria,
}: {
  active: boolean; onClick: () => void; children: React.ReactNode; aria: string
}) {
  return (
    <button
      onClick={onClick}
      aria-label={aria}
      aria-pressed={active}
      className="flex items-center justify-center transition-colors"
      style={{
        width: 22, height: 22, borderRadius: 4,
        background: active ? 'var(--tp-bg-elevated)' : 'transparent',
        color: active ? 'var(--tp-text)' : 'var(--tp-text-muted)',
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}
