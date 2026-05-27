/**
 * GlobalTopbar — 全局顶部 nav(PR-2,任务面板重置)。
 *
 * 复刻 spec mockup `docs/superpowers/specs/assets/2026-05-27-task-panel-reset/variant-final.html`
 * 的 .topbar 结构。布局:
 *
 *   nous-logo  · Workflow · Image · TTS · LLM · Models                    infra healthy · 3 GPU  ⌕  ⋮
 *   ─────────────────────── 5 服务 tab(主路由)───────────────────  ── 全局状态 / 搜索 / admin 下拉
 *
 * IconRail **保留**作为 admin 主 nav(D3 决策修正:用户偏好侧边栏继续放
 * Dashboard/API Key/Logs/Settings 等管理入口)。GlobalTopbar 右上 ⋮ 仅放
 * 「user 菜单」—— 当前只有退出登录(管理项请走 IconRail,避免冗余入口)。
 */
import { useState, useRef, useEffect } from 'react'
import {
  Search, MoreVertical, LogOut,
  GitBranch, Image as ImageIcon, Mic, MessageSquare, Database,
} from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
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

      {/* user 菜单(右上 ⋮)—— 只放退出登录;Dashboard/服务/API Key/用量/日志/设置/主题
          全在左侧 IconRail,避免冗余。登录态隐藏时整个按钮不显示(没东西可点)。 */}
      {canLogout && (
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((o) => !o)}
            className="p-1.5 ml-1 transition-colors"
            style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
            aria-label="用户菜单"
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
                minWidth: 160,
                background: 'var(--tp-bg-card)',
                border: '1px solid var(--tp-border-strong)',
                boxShadow: 'var(--shadow-card, 0 6px 16px rgba(0,0,0,0.3))',
              }}
            >
              <button
                onClick={handleLogout}
                role="menuitem"
                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors hover:bg-[var(--tp-bg-hover)]"
                style={{ color: 'var(--tp-text)', cursor: 'pointer' }}
              >
                <LogOut size={14} style={{ color: 'var(--tp-text-muted)' }} />
                退出登录
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
