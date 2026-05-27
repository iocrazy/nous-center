/**
 * GlobalTopbar — 全局顶部 nav(PR-2c 简化版,任务面板重置)。
 *
 * D7 决策修正:删原 5 服务 tab(Workflow/Image/TTS/LLM/Models)—— 用户心智模型里
 * 所有功能都通过 workflow 节点搭建(Image/TTS/LLM 是 workflow 类型而非独立路由)。
 * Topbar 现在只剩全局元素:
 *
 *   nous-logo                              infra healthy · 3 GPU  ⌕  ☰tasks  ⋮user
 *
 * 多 workflow 切换走 IconRail Workflow → 列表;Image/TTS/LLM 不再独立路由;
 * Models 通过 IconRail「引擎库」入口。
 */
import { useState, useRef, useEffect, useMemo } from 'react'
import {
  Search, MoreVertical, LogOut, ListTodo,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAdminLogout, useAdminMe } from '../../api/admin'
import { useTasks } from '../../api/tasks'
import { useExecutionStore } from '../../stores/execution'

export default function GlobalTopbar() {
  const navigate = useNavigate()
  const me = useAdminMe()
  const logout = useAdminLogout()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

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

      {/* PR-2b D4-2:任务管理入口 + badge + 下拉(D4-3 任务列表)。 */}
      <TaskMenu />

      {/* user 菜单(⋮)—— 只放退出登录(admin 项 + 主题在 IconRail);未登录态不渲染。 */}
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

const ACTIVE_STATUSES = new Set(['queued', 'running'])
const TASK_TYPE_LABEL: Record<string, string> = {
  image: '图像', tts: '语音', llm: '对话', vision: '视觉',
}
const STATUS_LABEL: Record<string, string> = {
  queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消',
}
const STATUS_COLOR: Record<string, string> = {
  queued: 'var(--status-queued)', running: 'var(--status-running)',
  completed: 'var(--tp-text-muted)', failed: 'var(--status-failed, #f87171)',
  cancelled: 'var(--tp-text-faint)',
}

/**
 * TaskMenu — GlobalTopbar 右侧任务管理入口(PR-2b D4-2/D4-3)。
 *
 * 显示当前活动任务数 badge,点击下拉「任务管理列表」—— 活动任务 + 最近完成 各 5 条。
 * 列表项点击可跳到对应任务详情(暂时 navigate 触发 IconRail TaskRailButton 同等动作 ——
 * 切到 panel 模式;PR-3 sidebar dock 上线后改为直接 highlight)。
 *
 * 实现复用 TaskMenuButton 的 useTasks() + useExecutionStore.toggleTaskPanel(),但把
 * 视觉从单按钮升级为 button + dropdown 列表(D4-3「下拉出 task 的任务管理列表」)。
 */
function TaskMenu() {
  const { data: tasks } = useTasks()
  const togglePanel = useExecutionStore((s) => s.toggleTaskPanel)
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  const { active, recent } = useMemo(() => {
    const all = tasks ?? []
    return {
      active: all.filter((t) => ACTIVE_STATUSES.has(t.status)).slice(0, 5),
      recent: all.filter((t) => !ACTIVE_STATUSES.has(t.status)).slice(0, 5),
    }
  }, [tasks])
  const badge = active.length

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  return (
    <div className="relative ml-1" ref={wrapRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={badge > 0 ? `${badge} 个活动任务` : '任务'}
        aria-expanded={open}
        aria-haspopup="menu"
        className="p-1.5 transition-colors relative"
        style={{
          color: badge > 0 ? 'var(--status-running)' : 'var(--tp-text-muted)',
          cursor: 'pointer',
        }}
      >
        <ListTodo size={16} />
        {badge > 0 && (
          <span
            className="absolute -top-0.5 -right-0.5 flex items-center justify-center text-[9px] font-semibold rounded-full"
            style={{
              minWidth: 14, height: 14, padding: '0 3px',
              background: 'var(--status-running)', color: '#0a0a0c',
            }}
          >
            {badge > 99 ? '99+' : badge}
          </span>
        )}
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-2 py-1 rounded-md"
          style={{
            width: 320, maxHeight: 480, overflowY: 'auto',
            background: 'var(--tp-bg-card)',
            border: '1px solid var(--tp-border-strong)',
            boxShadow: 'var(--shadow-card, 0 6px 16px rgba(0,0,0,0.3))',
          }}
        >
          <TaskGroup title={`活动任务 (${active.length})`} tasks={active} emptyText="当前无活动任务" />
          <div className="my-1" style={{ height: 1, background: 'var(--tp-border-faint)' }} />
          <TaskGroup title={`最近完成 (${recent.length})`} tasks={recent} emptyText="暂无历史记录" />
          <div className="my-1" style={{ height: 1, background: 'var(--tp-border-faint)' }} />
          <button
            onClick={() => { setOpen(false); togglePanel() }}
            role="menuitem"
            className="w-full text-center py-2 text-xs transition-colors hover:bg-[var(--tp-bg-hover)]"
            style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
          >
            打开完整任务面板 →
          </button>
        </div>
      )}
    </div>
  )
}

function TaskGroup({
  title, tasks, emptyText,
}: {
  title: string
  tasks: ReturnType<typeof useTasks>['data']
  emptyText: string
}) {
  const list = tasks ?? []
  return (
    <div>
      <div
        className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: 'var(--tp-text-muted)' }}
      >
        {title}
      </div>
      {list.length === 0 ? (
        <div className="px-3 py-2 text-xs" style={{ color: 'var(--tp-text-faint)' }}>{emptyText}</div>
      ) : (
        list.map((t) => (
          <div
            key={t.id}
            className="flex items-center gap-2 px-3 py-1.5 text-xs"
            style={{ color: 'var(--tp-text)' }}
          >
            <span
              style={{
                width: 7, height: 7, borderRadius: '50%',
                background: STATUS_COLOR[t.status] ?? 'var(--tp-text-faint)',
                boxShadow: t.status === 'running' ? '0 0 8px var(--status-running)' : undefined,
                flexShrink: 0,
              }}
              title={STATUS_LABEL[t.status] ?? t.status}
            />
            <span className="truncate flex-1" title={t.workflow_name || `#${t.id}`}>
              {t.workflow_name || `#${t.id}`}
            </span>
            {t.task_type && TASK_TYPE_LABEL[t.task_type] && (
              <span
                className="text-[9px] px-1.5 py-0.5 rounded shrink-0"
                style={{
                  background: 'var(--tp-bg-elevated)', color: 'var(--tp-text-muted)',
                }}
              >
                {TASK_TYPE_LABEL[t.task_type]}
              </span>
            )}
            <span className="text-[10px] shrink-0" style={{ color: 'var(--tp-text-muted)' }}>
              {STATUS_LABEL[t.status] ?? t.status}
            </span>
          </div>
        ))
      )}
    </div>
  )
}
