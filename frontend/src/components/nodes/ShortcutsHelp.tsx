import { useState, useEffect } from 'react'
import { Keyboard, X } from 'lucide-react'

// 画布快捷键帮助面板(对齐 Infinite-Canvas smartShortcutToggle)。
// 触发:右下角 keyboard 按钮 或 按 `?`(Shift+/);Esc 关闭。
// 数据驱动,新增快捷键只改 SHORTCUTS。

interface ShortcutRow {
  keys: string[]
  desc: string
}

const SHORTCUTS: { group: string; rows: ShortcutRow[] }[] = [
  {
    group: '选择 / 移动',
    rows: [
      { keys: ['Ctrl', '拖拽'], desc: '框选多个节点' },
      { keys: ['Ctrl', '点击'], desc: '加选 / 取消选中' },
      { keys: ['拖拽空白'], desc: '平移画布' },
      { keys: ['滚轮'], desc: '缩放画布' },
    ],
  },
  {
    group: '编辑',
    rows: [
      { keys: ['Ctrl', 'C'], desc: '复制选中节点' },
      { keys: ['Ctrl', 'V'], desc: '粘贴' },
      { keys: ['Ctrl', 'D'], desc: '原地复制' },
      { keys: ['Alt', '拖拽'], desc: '拖拽复制节点' },
      { keys: ['Ctrl', 'B'], desc: '旁路 / 取消旁路' },
      { keys: ['Del'], desc: '删除选中节点' },
      { keys: ['Ctrl', 'Z'], desc: '撤销' },
      { keys: ['Ctrl', 'Shift', 'Z'], desc: '重做' },
    ],
  },
  {
    group: '分组',
    rows: [
      { keys: ['Ctrl', 'G'], desc: '成组' },
      { keys: ['Ctrl', 'Shift', 'G'], desc: '解组(不删节点)' },
    ],
  },
  {
    group: '画布 / 节点',
    rows: [
      { keys: ['Z'], desc: '缩放到全部节点总览' },
      { keys: ['双击空白'], desc: '打开建节点菜单' },
      { keys: ['右键空白'], desc: '建节点菜单' },
      { keys: ['拖端口到空白'], desc: '建相连节点' },
    ],
  },
]

function isEditableTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null
  if (!el) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable
}

export default function ShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setOpen(false); return }
      // `?`(Shift+/)切换;编辑态放行。
      if (e.key === '?' && !isEditableTarget(e.target)) {
        e.preventDefault()
        setOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <>
      {/* 触发按钮(右下,minimap 上方) */}
      <button
        type="button"
        title="快捷键 (?)"
        aria-label="快捷键"
        onClick={() => setOpen(true)}
        style={{
          position: 'absolute', bottom: 214, right: 14, zIndex: 6,
          width: 30, height: 30, borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--card)', border: '1px solid var(--border)',
          color: 'var(--muted)', cursor: 'pointer', boxShadow: 'var(--shadow-md)',
        }}
      >
        <Keyboard size={15} />
      </button>

      {open && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position: 'absolute', inset: 0, zIndex: 50,
            background: 'rgba(0,0,0,0.4)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 460, maxHeight: '80%', overflow: 'auto',
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
              borderRadius: 'var(--node-radius, 14px)', boxShadow: 'var(--shadow-lg)',
              padding: 18,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <Keyboard size={16} style={{ color: 'var(--text)' }} />
              <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-strong)', flex: 1 }}>快捷键</span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="关闭"
                style={{ display: 'flex', padding: 4, background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}
              >
                <X size={16} />
              </button>
            </div>

            {SHORTCUTS.map((sec) => (
              <div key={sec.group} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--muted)', marginBottom: 6 }}>
                  {sec.group}
                </div>
                {sec.rows.map((r, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
                    <span style={{ display: 'flex', gap: 4, flexShrink: 0, minWidth: 150 }}>
                      {r.keys.map((k, j) => (
                        <kbd
                          key={j}
                          style={{
                            fontSize: 10, fontFamily: 'var(--mono)', padding: '2px 6px', borderRadius: 5,
                            background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--text)',
                          }}
                        >
                          {k}
                        </kbd>
                      ))}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text)' }}>{r.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}
