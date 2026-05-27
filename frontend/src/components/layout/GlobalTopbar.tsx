/**
 * GlobalTopbar — 全局顶部 nav(PR-3a 任务面板重置)。
 *
 * D10 决策修正:task UI 是 **GlobalTopbar 下拉悬浮面板**(不是右侧固定 sidebar)。
 * 下拉里复刻 mockup `variant-final.html` 的完整 task 结构:Active/History tabs +
 * cook overview(运行/排队/完成 + GPU)+ 任务行(4 type 配色)。
 *
 * Topbar 布局:
 *   nous-logo  · spacer · infra healthy · ⌕ search · ☰ tasks(下拉)· ⋮ user
 */
import { useState, useRef, useEffect, useMemo } from 'react'
import { Search, MoreVertical, LogOut, ListTodo, Cpu } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAdminLogout, useAdminMe } from '../../api/admin'
import { useTasks, type ExecutionTask } from '../../api/tasks'
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

      {/* 搜索 placeholder */}
      <button
        className="p-1.5 transition-colors"
        style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
        aria-label="搜索"
        title="搜索(占位)"
      >
        <Search size={16} />
      </button>

      {/* PR-3a:任务下拉悬浮面板(对齐 mockup variant-final sidebar 形态)*/}
      <TaskMenu />

      {/* user 菜单(⋮)—— 只放退出登录。 */}
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

/* ============================================================
 * TaskMenu — 任务下拉悬浮面板(PR-3a)
 *
 * 入口:Topbar 右侧 ListTodo 图标 + 活动任务 badge。
 * 下拉内容(复刻 mockup variant-final sidebar 内部结构,但作为悬浮 panel):
 *   ┌─ sb-header:全部任务 [global queue]
 *   ├─ sb-tabs:Active N · History N
 *   ├─ cook-overview:●N 运行 · ●N 排队 · ●N 完成 · GPU
 *   ├─ sb-body:Active tab → 任务行(timeline node + task-card + L3 callout)
 *   │           History tab → 4 type 任务行(image/tts/llm/vision 配色)
 *   └─ footer:打开完整任务面板 →
 * ============================================================ */

const ACTIVE_STATUSES = new Set(['queued', 'running'])
const TASK_TYPES = ['image', 'tts', 'vision', 'llm'] as const
type TaskType = typeof TASK_TYPES[number]

const TASK_TYPE_LABEL: Record<TaskType, string> = {
  image: 'IMAGE', tts: 'TTS', vision: 'VISION', llm: 'LLM',
}
const STATUS_LABEL: Record<string, string> = {
  queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消',
}
const STATUS_COLOR: Record<string, string> = {
  queued: 'var(--status-queued)', running: 'var(--status-running)',
  completed: 'var(--tp-text-muted)', failed: 'var(--status-failed, #f87171)',
  cancelled: 'var(--tp-text-faint)',
}

function getTaskType(t: ExecutionTask): TaskType | null {
  const v = (t as ExecutionTask & { type?: string }).type ?? t.task_type
  return v === 'image' || v === 'tts' || v === 'vision' || v === 'llm' ? v : null
}

function TaskMenu() {
  const { data: tasks } = useTasks()
  const togglePanel = useExecutionStore((s) => s.toggleTaskPanel)
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'active' | 'history'>('active')
  const wrapRef = useRef<HTMLDivElement>(null)

  const counts = useMemo(() => {
    const all = tasks ?? []
    const running = all.filter((t) => t.status === 'running').length
    const queued = all.filter((t) => t.status === 'queued').length
    const done = all.filter((t) => t.status === 'completed').length
    const active = running + queued
    const history = all.length - active
    return { running, queued, done, active, history }
  }, [tasks])

  const { activeTasks, historyTasks } = useMemo(() => {
    const all = tasks ?? []
    return {
      activeTasks: all.filter((t) => ACTIVE_STATUSES.has(t.status)),
      historyTasks: all.filter((t) => !ACTIVE_STATUSES.has(t.status)),
    }
  }, [tasks])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const badge = counts.active

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
          className="absolute right-0 top-full mt-2 rounded-md overflow-hidden flex flex-col"
          style={{
            width: 400, maxHeight: 600,
            background: 'var(--tp-bg-panel)',
            border: '1px solid var(--tp-border-strong)',
            boxShadow: 'var(--shadow-card, 0 12px 32px rgba(0,0,0,0.5))',
          }}
        >
          {/* sb-header */}
          <div
            className="flex items-center gap-2 shrink-0"
            style={{ height: 44, padding: '0 16px', borderBottom: '1px solid var(--tp-border-faint)' }}
          >
            <span className="text-sm font-semibold" style={{ color: 'var(--tp-text)' }}>
              全部任务
            </span>
            <span
              className="text-[10px] font-medium px-1.5 py-0.5 rounded font-mono"
              style={{
                background: 'var(--tp-bg-elevated)',
                color: 'var(--tp-text-muted)',
              }}
            >
              global queue
            </span>
          </div>

          {/* sb-tabs */}
          <div
            className="flex shrink-0"
            style={{ padding: '0 16px', gap: 18, borderBottom: '1px solid var(--tp-border-faint)' }}
          >
            <SbTab label="Active" count={counts.active} active={tab === 'active'} onClick={() => setTab('active')} />
            <SbTab label="History" count={counts.history} active={tab === 'history'} onClick={() => setTab('history')} />
          </div>

          {/* cook-overview */}
          <div
            className="shrink-0"
            style={{
              padding: '10px 16px',
              background: 'linear-gradient(180deg, var(--tp-bg-card-subtle, #0f0f12) 0%, var(--tp-bg-panel) 100%)',
              borderBottom: '1px solid var(--tp-border-faint)',
            }}
          >
            <div className="flex items-center gap-3.5 text-[11.5px]">
              <CookStat dotColor="var(--status-running)" glow num={counts.running} label="运行" />
              <CookStat dotColor="var(--status-queued)" num={counts.queued} label="排队" />
              <CookStat dotColor="var(--tp-text-dim)" num={counts.done} label="完成" />
              <span
                className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-mono"
                style={{ color: 'var(--tp-text-muted)' }}
              >
                <Cpu size={12} style={{ color: 'var(--status-running)' }} />
                GPU
              </span>
            </div>
          </div>

          {/* sb-body */}
          <div className="flex-1 overflow-y-auto">
            {tab === 'active' ? (
              <TaskList tasks={activeTasks} emptyText="当前无活动任务" sectionLabel={`正在跑 · ${counts.active}`} />
            ) : (
              <TaskList tasks={historyTasks} emptyText="暂无历史记录" sectionLabel={`历史 · ${counts.history}`} />
            )}
          </div>

          <div className="shrink-0" style={{ borderTop: '1px solid var(--tp-border-faint)' }}>
            <button
              onClick={() => { setOpen(false); togglePanel() }}
              role="menuitem"
              className="w-full text-center py-2.5 text-xs transition-colors hover:bg-[var(--tp-bg-hover)]"
              style={{ color: 'var(--tp-text-muted)', cursor: 'pointer' }}
            >
              打开完整任务面板 →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function SbTab({
  label, count, active, onClick,
}: {
  label: string
  count: number
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 text-xs font-medium transition-colors"
      style={{
        height: 36,
        color: active ? 'var(--tp-text)' : 'var(--tp-text-muted)',
        borderBottom: `2px solid ${active ? 'var(--status-running)' : 'transparent'}`,
        cursor: 'pointer',
      }}
      aria-current={active ? 'page' : undefined}
    >
      {label}
      <span
        className="text-[10px] px-1.5 py-0.5 rounded font-mono"
        style={{
          background: active ? 'rgba(74, 222, 128, 0.13)' : 'var(--tp-bg-elevated)',
          color: active ? 'var(--status-running)' : 'var(--tp-text-muted)',
        }}
      >
        {count}
      </span>
    </button>
  )
}

function CookStat({
  dotColor, glow, num, label,
}: {
  dotColor: string
  glow?: boolean
  num: number
  label: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5" style={{ color: 'var(--tp-text)' }}>
      <span
        style={{
          width: 7, height: 7, borderRadius: '50%',
          background: dotColor,
          boxShadow: glow ? `0 0 8px ${dotColor}` : undefined,
          flexShrink: 0,
        }}
      />
      <span className="font-mono font-semibold">{num}</span>
      <span style={{ color: 'var(--tp-text-muted)' }}>{label}</span>
    </span>
  )
}

function TaskList({
  tasks, emptyText, sectionLabel,
}: {
  tasks: ExecutionTask[]
  emptyText: string
  sectionLabel: string
}) {
  return (
    <div className="p-3">
      <div
        className="text-[10px] uppercase tracking-wider font-semibold mb-2 px-1"
        style={{ color: 'var(--tp-text-muted)' }}
      >
        {sectionLabel}
      </div>
      {tasks.length === 0 ? (
        <div className="text-xs py-3 text-center" style={{ color: 'var(--tp-text-faint)' }}>
          {emptyText}
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          {tasks.slice(0, 20).map((t) => <TaskRow key={t.id} task={t} />)}
        </div>
      )}
    </div>
  )
}

/**
 * TaskRow — 任务卡片(对齐 mockup task-card 紧凑版)。
 *
 * 行结构:[type chip 4 色] [workflow_name] [status chip] [meta]
 * type chip 按 image 紫 / tts 青 / llm 蓝 / vision 橙 配色。
 * status running 时左侧加 glow dot;done 状态灰 chip。
 */
function TaskRow({ task: t }: { task: ExecutionTask }) {
  const type = getTaskType(t)
  const isRunning = t.status === 'running'
  return (
    <div
      className="flex items-center gap-2 p-2 rounded transition-colors hover:bg-[var(--tp-bg-hover)]"
      style={{
        background: isRunning ? 'rgba(74, 222, 128, 0.05)' : 'var(--tp-bg-card)',
        border: `1px solid ${isRunning ? 'rgba(74, 222, 128, 0.4)' : 'var(--tp-border)'}`,
      }}
    >
      {type && (
        <span
          className="text-[9px] font-bold tracking-wider font-mono px-1.5 py-0.5 rounded shrink-0"
          style={{
            background: `var(--type-${type}-bg-chip)`,
            color: `var(--type-${type})`,
          }}
          title={TASK_TYPE_LABEL[type]}
        >
          {TASK_TYPE_LABEL[type]}
        </span>
      )}
      <span
        className="text-xs font-mono truncate flex-1"
        style={{ color: 'var(--tp-text)' }}
        title={t.workflow_name || `#${t.id}`}
      >
        {t.workflow_name || `#${t.id}`}
      </span>
      <span
        className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded shrink-0 inline-flex items-center gap-1"
        style={{
          background: isRunning ? 'rgba(74, 222, 128, 0.13)' : 'var(--tp-bg-elevated)',
          color: STATUS_COLOR[t.status] ?? 'var(--tp-text-muted)',
        }}
      >
        {isRunning && (
          <span
            style={{
              width: 5, height: 5, borderRadius: '50%',
              background: 'var(--status-running)',
              boxShadow: '0 0 6px var(--status-running)',
            }}
          />
        )}
        {STATUS_LABEL[t.status] ?? t.status}
      </span>
    </div>
  )
}
