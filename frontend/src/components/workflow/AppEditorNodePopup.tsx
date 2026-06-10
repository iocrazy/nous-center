// 节点属性编辑弹窗(spec 2026-06-10,复刻 Infinite-Canvas openNodePopup)。
// 点画布节点 → 弹出此窗,列出该节点所有可暴露参数,每行:勾选框 + 默认值徽章 +
// 改名输入 + 控件类型下拉。勾选/改名/改类型即时回写,左侧表单实时预览。
import { useEffect } from 'react'
import { X } from 'lucide-react'
import type { ExposableRow } from './appEditorSchema'
import { paramId } from './appEditorSchema'
import type { ExposedParam } from '../../api/services'
import { KIND_OPTIONS, kindOf, type WidgetKind } from './widgetKind'
import NodeSelectPopover from '../nodes/NodeSelectPopover'

export interface AppEditorNodePopupProps {
  title: string
  sub: string
  rows: ExposableRow[]
  /** 该节点当前已暴露的参数,按 paramId 索引(决定勾选态/显示名/类型)。 */
  exposed: Map<string, ExposedParam>
  onToggle: (inputName: string) => void
  onRename: (inputName: string, label: string) => void
  onChangeKind: (inputName: string, kind: WidgetKind) => void
  onClose: () => void
}

function ValueBadge({ value }: { value: unknown }) {
  if (typeof value === 'string') {
    const v = value.length > 50 ? value.slice(0, 50) + '…' : value
    return <span style={{ color: 'var(--text)', fontWeight: 700 }}>"{v}"</span>
  }
  if (typeof value === 'number') {
    return <span style={{ color: '#0369a1', fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
  }
  if (typeof value === 'boolean') {
    return <span style={{ color: value ? '#15803d' : '#b45309', fontWeight: 800 }}>{value ? '✓ true' : '✗ false'}</span>
  }
  if (value == null) return <span style={{ color: 'var(--muted)' }}>—</span>
  return <span style={{ color: 'var(--muted)' }}>{String(value)}</span>
}

export default function AppEditorNodePopup({
  title, sub, rows, exposed, onToggle, onRename, onChangeKind, onClose,
}: AppEditorNodePopupProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const smallInput: React.CSSProperties = {
    width: '100%', fontSize: 12, padding: '6px 8px',
    background: 'var(--bg)', color: 'var(--text)',
    border: '1px solid var(--border)', borderRadius: 6,
  }

  return (
    <>
      <div
        onClick={onClose}
        style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.28)', zIndex: 20 }}
      />
      <div
        role="dialog"
        style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
          width: 420, maxWidth: 'calc(100% - 32px)', maxHeight: 'calc(100% - 48px)',
          display: 'flex', flexDirection: 'column', zIndex: 21,
          background: 'var(--card)', border: '1px solid var(--border)',
          borderRadius: 14, boxShadow: 'var(--shadow-lg, 0 12px 40px rgba(0,0,0,0.25))',
          overflow: 'hidden',
        }}
      >
        {/* head */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px',
          borderBottom: '1px solid var(--border)', background: 'var(--card-hl)',
        }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{title}</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sub}</div>
          </div>
          <button
            type="button" title="关闭" onClick={onClose}
            style={{ display: 'inline-flex', padding: 4, background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer', borderRadius: 6 }}
          >
            <X size={16} />
          </button>
        </div>

        {/* body */}
        <div style={{ overflow: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {rows.length === 0 && (
            <div style={{ color: 'var(--muted)', fontSize: 12, padding: '12px 4px' }}>该节点无可配置字段</div>
          )}
          {rows.map((r) => {
            const cur = exposed.get(paramId(r.param))
            const active = !!cur
            const showKey = r.label !== r.input_name
            const kind = kindOf(cur ?? r.param)
            return (
              <div
                key={r.input_name}
                style={{
                  border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 10, padding: 10,
                  background: active ? 'var(--accent-subtle, rgba(99,102,241,0.06))' : 'transparent',
                  display: 'flex', flexDirection: 'column', gap: 8,
                }}
              >
                <button
                  type="button"
                  onClick={() => onToggle(r.input_name)}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 9, width: '100%',
                    background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, textAlign: 'left',
                  }}
                >
                  <span style={{
                    width: 17, height: 17, marginTop: 1, flexShrink: 0, borderRadius: 5,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: active ? 'var(--accent)' : 'transparent',
                    border: `1.5px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                  }}>
                    {active && <span style={{ color: '#fff', fontSize: 11, fontWeight: 900, lineHeight: 1 }}>✓</span>}
                  </span>
                  <span style={{ minWidth: 0, flex: 1 }}>
                    <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>
                      {r.label}
                      {showKey && <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--faint, var(--muted))', marginLeft: 5 }}>{r.input_name}</span>}
                    </span>
                    <span style={{ display: 'block', fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
                      默认值: <ValueBadge value={r.param.default} />
                    </span>
                  </span>
                </button>

                {/* 显示名 + 控件类型(仅勾选后可编辑) */}
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    style={{ ...smallInput, flex: 1, opacity: active ? 1 : 0.5 }}
                    placeholder="显示名"
                    value={(cur?.label ?? r.label) || ''}
                    disabled={!active}
                    onChange={(e) => onRename(r.input_name, e.target.value)}
                  />
                  {/* 控件类型:用全局 NodeSelectPopover(对齐画布节点下拉样式),非勾选态置灰静态 */}
                  {active ? (
                    <NodeSelectPopover
                      value={kind}
                      onChange={(v) => onChangeKind(r.input_name, v as WidgetKind)}
                      options={KIND_OPTIONS.map((o) => ({ value: o.v, label: o.label }))}
                      size="compact"
                      width={110}
                    />
                  ) : (
                    <div style={{ ...smallInput, width: 110, flexShrink: 0, opacity: 0.5, color: 'var(--muted)', display: 'flex', alignItems: 'center' }}>
                      {KIND_OPTIONS.find((o) => o.v === kind)?.label ?? kind}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
