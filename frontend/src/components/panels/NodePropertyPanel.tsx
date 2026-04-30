import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronRight } from 'lucide-react'
import { useSelectionStore } from '../../stores/selection'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import { DECLARATIVE_NODES, type WidgetDef } from '../../models/nodeRegistry'
import { useAgents } from '../../api/agents'
import { apiFetch } from '../../api/client'
import { useEnginesLiveSync, type EngineInfo } from '../../api/engines'

// m09 v3: 右侧节点属性面板。选中节点 → 渲染该节点的 widgets，
// 跟节点体内的 widget 共享 useWorkspaceStore.updateNode，所以两边
// 永远同步（不会出现"面板改了节点没动"）。
//
// 节点体内的小 widget 仍保留 — 拖拽时可以快速看 + 改；面板提供
// 大空间编辑长 prompt / 复杂参数。

export default function NodePropertyPanel() {
  const selectedNodeId = useSelectionStore((s) => s.selectedNodeId)
  const workflow = useWorkspaceStore((s) => s.getActiveWorkflow())
  const updateNode = useWorkspaceStore((s) => s.updateNode)

  const node = useMemo(
    () => workflow.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [workflow.nodes, selectedNodeId],
  )

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        width: 300,
        background: 'var(--bg-accent)',
        borderLeft: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10,
        overflow: 'hidden',
      }}
    >
      {!node ? (
        <EmptyState />
      ) : (
        <PropertyView
          nodeId={node.id}
          nodeType={String(node.type)}
          data={node.data ?? {}}
          onChange={(patch) => updateNode(node.id, patch)}
        />
      )}
    </div>
  )
}

// ---------- subviews ----------

function EmptyState() {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 24,
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontSize: 12,
          color: 'var(--muted)',
          fontWeight: 500,
          marginBottom: 6,
        }}
      >
        节点属性
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}>
        点击画布上的一个节点
        <br />
        在这里编辑它的字段
      </div>
    </div>
  )
}

function PropertyView({
  nodeId,
  nodeType,
  data,
  onChange,
}: {
  nodeId: string
  nodeType: string
  data: Record<string, unknown>
  onChange: (patch: Record<string, unknown>) => void
}) {
  const declDef = DECLARATIVE_NODES[nodeType]
  const portDef = NODE_DEFS[nodeType]
  const label = declDef?.label ?? portDef?.label ?? nodeType

  return (
    <>
      {/* header */}
      <div
        style={{
          padding: '12px 14px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {declDef && (
            <span
              style={{
                fontSize: 9,
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: 3,
                background: `color-mix(in srgb, ${declDef.badgeColor} 15%, transparent)`,
                color: declDef.badgeColor,
                letterSpacing: 0.5,
              }}
            >
              {declDef.badge}
            </span>
          )}
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
            {label}
          </span>
        </div>
        <div
          style={{
            fontSize: 10,
            color: 'var(--muted)',
            marginTop: 4,
            fontFamily: 'var(--mono, monospace)',
          }}
        >
          id: {nodeId}
        </div>
      </div>

      {/* body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        {!declDef ? (
          <PortsOnlyView portDef={portDef} />
        ) : declDef.widgets.length === 0 ? (
          <div style={{ padding: 14, fontSize: 11, color: 'var(--muted)' }}>
            该节点无可编辑字段。
          </div>
        ) : (
          declDef.widgets.map((w) => (
            <FieldRow key={w.name} widget={w}>
              <FieldRenderer
                widget={w}
                value={data[w.name]}
                onChange={(v) => onChange({ [w.name]: v })}
              />
            </FieldRow>
          ))
        )}

        {portDef && (portDef.inputs.length > 0 || portDef.outputs.length > 0) && (
          <PortsSection portDef={portDef} />
        )}
      </div>
    </>
  )
}

function FieldRow({ widget, children }: { widget: WidgetDef; children: React.ReactNode }) {
  return (
    <div style={{ padding: '8px 14px' }}>
      <label
        style={{
          display: 'block',
          fontSize: 10,
          color: 'var(--muted)',
          marginBottom: 4,
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          fontWeight: 600,
        }}
      >
        {widget.label}
      </label>
      {children}
    </div>
  )
}

function FieldRenderer({
  widget,
  value,
  onChange,
}: {
  widget: WidgetDef
  value: unknown
  onChange: (v: unknown) => void
}) {
  const v = value !== undefined && value !== null ? value : widget.default

  switch (widget.widget) {
    case 'input':
      return (
        <input
          value={String(v ?? '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={widget.label}
          style={inputStyle}
        />
      )
    case 'textarea':
      return (
        <textarea
          value={String(v ?? '')}
          onChange={(e) => onChange(e.target.value)}
          rows={Math.max(widget.rows ?? 3, 3)}
          style={{
            ...inputStyle,
            resize: 'vertical',
            fontFamily: 'var(--mono, monospace)',
            minHeight: 60,
          }}
        />
      )
    case 'select':
      return (
        <select
          value={String(v ?? '')}
          onChange={(e) => onChange(e.target.value)}
          style={inputStyle}
        >
          {widget.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      )
    case 'slider':
      return <SliderField widget={widget} value={Number(v ?? widget.min ?? 0)} onChange={onChange} />
    case 'checkbox':
      return (
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={!!v}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span style={{ fontSize: 11, color: 'var(--text)' }}>{widget.label}</span>
        </label>
      )
    case 'agent_select':
      return <AgentSelect value={String(v ?? '')} onChange={onChange} />
    case 'model_select':
      return (
        <ModelSelect value={String(v ?? '')} onChange={onChange} filter={widget.filter} />
      )
    default:
      return <span style={{ fontSize: 11, color: 'var(--muted)' }}>—</span>
  }
}

function SliderField({
  widget,
  value,
  onChange,
}: {
  widget: WidgetDef
  value: number
  onChange: (v: number) => void
}) {
  const min = widget.min ?? 0
  const max = widget.max ?? 1
  const step = widget.step ?? 1
  const precision = widget.precision ?? 0
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1 }}
      />
      <input
        type="number"
        value={precision > 0 ? value.toFixed(precision) : value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ...inputStyle, width: 70 }}
      />
    </div>
  )
}

function AgentSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data: agents } = useAgents()
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={inputStyle}
    >
      <option value="">选择 Agent...</option>
      {agents?.map((a) => (
        <option key={a.name} value={a.name}>
          {a.display_name || a.name}
        </option>
      ))}
    </select>
  )
}

function ModelSelect({
  value,
  onChange,
  filter,
}: {
  value: string
  onChange: (v: string) => void
  filter?: string
}) {
  useEnginesLiveSync()
  const params = filter ? `?type=${filter}` : ''
  const { data: engines } = useQuery({
    queryKey: ['engines', filter],
    queryFn: () => apiFetch<EngineInfo[]>(`/api/v1/engines${params}`),
  })
  const loaded = (engines ?? []).filter((e) => e.status === 'loaded')
  const unloaded = (engines ?? []).filter((e) => e.status !== 'loaded' && e.local_exists)
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={inputStyle}
    >
      <option value="">选择模型...</option>
      {loaded.map((e) => (
        <option key={e.name} value={e.name}>
          {e.display_name}
        </option>
      ))}
      {unloaded.length > 0 && loaded.length > 0 && <option disabled>──── 未加载 ────</option>}
      {unloaded.map((e) => (
        <option key={e.name} value={e.name} disabled>
          {e.display_name} (未加载)
        </option>
      ))}
    </select>
  )
}

function PortsOnlyView({ portDef }: { portDef?: { type: string; label: string } }) {
  return (
    <div style={{ padding: 14, fontSize: 11, color: 'var(--muted)' }}>
      {portDef?.label ?? '内置节点'}（无可编辑参数）
    </div>
  )
}

function PortsSection({
  portDef,
}: {
  portDef: { inputs: { id: string; type: string; label: string }[]; outputs: { id: string; type: string; label: string }[] }
}) {
  return (
    <div
      style={{
        marginTop: 6,
        borderTop: '1px solid var(--border)',
        padding: '10px 14px',
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 600,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          marginBottom: 8,
        }}
      >
        端口
      </div>
      {portDef.inputs.length > 0 && (
        <PortGroup label="输入" ports={portDef.inputs} />
      )}
      {portDef.outputs.length > 0 && (
        <PortGroup label="输出" ports={portDef.outputs} />
      )}
    </div>
  )
}

function PortGroup({
  label,
  ports,
}: {
  label: string
  ports: { id: string; type: string; label: string }[]
}) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
      {ports.map((p) => (
        <div
          key={p.id}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 11,
            padding: '3px 0',
            color: 'var(--text)',
          }}
        >
          <ChevronRight size={10} style={{ color: 'var(--muted)' }} />
          <span>{p.label}</span>
          <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--muted)' }}>{p.type}</span>
        </div>
      ))}
    </div>
  )
}

const inputStyle = {
  width: '100%',
  background: 'var(--bg)',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '6px 8px',
  fontSize: 12,
  outline: 'none',
} as const
