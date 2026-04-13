import { useState, useRef, useCallback, useEffect } from 'react'
import { Handle, Position, useUpdateNodeInternals, useReactFlow, useNodeId } from '@xyflow/react'
import type { PortDef } from '../../models/workflow'

const PORT_TYPE_COLORS: Record<string, string> = {
  text: 'var(--ok)',
  audio: 'var(--info)',
  control: 'var(--accent)',
  any: 'var(--purple)',
}

interface BaseNodeProps {
  title: string
  badge?: { label: string; color: string; bg: string }
  selected?: boolean
  inputs: PortDef[]
  outputs: PortDef[]
  children?: React.ReactNode
}

export default function BaseNode({ title, badge, selected, inputs, outputs, children }: BaseNodeProps) {
  const [collapsed, setCollapsed] = useState(false)
  const nodeId = useNodeId()
  const updateNodeInternals = useUpdateNodeInternals()
  const { setNodes } = useReactFlow()

  // When collapsing: clear node height so wrapper auto-sizes
  // When expanding: restore height so resize works
  useEffect(() => {
    if (!nodeId) return
    if (collapsed) {
      // Remove height constraints — let wrapper shrink to content
      setNodes((nds) => nds.map((n) =>
        n.id === nodeId
          ? { ...n, height: undefined, style: { ...n.style, height: undefined } }
          : n
      ))
    }
    const t = setTimeout(() => updateNodeInternals(nodeId), 50)
    return () => clearTimeout(t)
  }, [collapsed, nodeId, updateNodeInternals, setNodes])

  // Build rows: merge input + output ports on same logical row index
  const maxPorts = Math.max(inputs.length, outputs.length)
  const portRows: { input?: PortDef; output?: PortDef }[] = []
  for (let i = 0; i < maxPorts; i++) {
    portRows.push({ input: inputs[i], output: outputs[i] })
  }

  return (
    <div
      style={{
        minWidth: 180,
        width: '100%',
        minHeight: collapsed ? undefined : '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--card)',
        border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 8,
        boxShadow: selected
          ? 'var(--shadow-md), 0 0 0 2px var(--accent-subtle)'
          : 'var(--shadow-md)',
      }}
    >
      {/* Header */}
      <div
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          padding: '6px 10px',
          borderBottom: collapsed ? 'none' : '1px solid var(--border)',
          background: 'var(--card-hl)',
          borderRadius: collapsed ? 8 : '8px 8px 0 0',
          cursor: 'pointer',
          userSelect: 'none',
          position: 'relative',
        }}
      >
        <span
          style={{
            fontSize: 8,
            color: 'var(--muted)',
            width: 10,
            transition: 'transform 0.15s',
            transform: collapsed ? 'rotate(-90deg)' : 'none',
          }}
        >
          &#9660;
        </span>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-strong)' }}>{title}</span>
        {badge && (
          <span
            className="ml-auto"
            style={{
              fontSize: 8,
              padding: '1px 5px',
              borderRadius: 3,
              fontWeight: 600,
              background: badge.bg,
              color: badge.color,
            }}
          >
            {badge.label}
          </span>
        )}
      </div>

      {/* Port rows — always rendered so Handle positions stay consistent */}
      {portRows.map((row, i) => (
        <div
          key={i}
          style={collapsed ? {
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            pointerEvents: 'none',
          } : {
            display: 'flex',
            alignItems: 'center',
            padding: '4px 10px',
            position: 'relative',
            minHeight: 24,
            borderBottom: '1px solid rgba(255,255,255,0.02)',
          }}
        >
          {row.input && (
            <>
              <Handle
                type="target"
                position={Position.Left}
                id={row.input.id}
                style={{
                  background: PORT_TYPE_COLORS[row.input.type] ?? 'var(--muted)',
                  width: 10,
                  height: 10,
                  border: '2px solid var(--card)',
                  left: -5,
                }}
              />
              {!collapsed && (
                <span style={{ fontSize: 10, color: 'var(--muted)', flexShrink: 0 }}>
                  {row.input.label}
                </span>
              )}
            </>
          )}
          {row.output && (
            <>
              {!collapsed && (
                <span
                  style={{
                    fontSize: 10,
                    color: 'var(--muted)',
                    marginLeft: 'auto',
                    textAlign: 'right',
                    flexShrink: 0,
                  }}
                >
                  {row.output.label}
                </span>
              )}
              <Handle
                type="source"
                position={Position.Right}
                id={row.output.id}
                style={{
                  background: PORT_TYPE_COLORS[row.output.type] ?? 'var(--muted)',
                  width: 10,
                  height: 10,
                  border: '2px solid var(--card)',
                  right: -5,
                }}
              />
            </>
          )}
        </div>
      ))}

      {/* Widget children — hidden when collapsed */}
      {!collapsed && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {children}
        </div>
      )}
    </div>
  )
}

/** A widget row inside a node (label + control).
 *  stretch=true makes this row expand to fill available vertical space (for textareas).
 */
export function NodeWidgetRow({
  label,
  children,
  stretch,
}: {
  label: string
  children: React.ReactNode
  stretch?: boolean
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: stretch ? 'stretch' : 'center',
        flexDirection: stretch ? 'column' : 'row',
        padding: '4px 10px',
        position: 'relative',
        minHeight: stretch ? 48 : 24,
        borderBottom: '1px solid rgba(255,255,255,0.02)',
        gap: stretch ? 2 : 6,
        ...(stretch ? { flex: 1, minHeight: 0 } : {}),
      }}
    >
      <span style={{ fontSize: 10, color: 'var(--muted)', flexShrink: 0, ...(stretch ? {} : { width: 110 }), textAlign: 'left' }}>{label}</span>
      <div style={{ flex: 1, minWidth: 0, minHeight: 0, display: 'flex', flexDirection: 'column' }}>{children}</div>
    </div>
  )
}

/** Styled inline widget input — "nodrag" prevents React Flow node drag */
export function NodeInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className="nodrag"
      type="text"
      style={{
        width: '100%',
        padding: '3px 7px',
        fontSize: 10,
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 3,
        color: 'var(--text)',
        fontFamily: 'var(--font)',
        outline: 'none',
        ...props.style,
      }}
    />
  )
}

/** Styled inline widget select with custom dropdown arrow */
export function NodeSelect(props: React.SelectHTMLAttributes<HTMLSelectElement> & { children: React.ReactNode }) {
  return (
    <div className="nodrag" style={{ position: 'relative', width: '100%' }}>
      <select
        {...props}
        style={{
          width: '100%',
          padding: '3px 20px 3px 7px',
          fontSize: 10,
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 3,
          color: 'var(--text)',
          fontFamily: 'var(--font)',
          outline: 'none',
          appearance: 'none',
          WebkitAppearance: 'none',
          cursor: 'pointer',
          ...props.style,
        }}
      />
      <svg
        width="8"
        height="5"
        viewBox="0 0 8 5"
        style={{
          position: 'absolute',
          right: 6,
          top: '50%',
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
        }}
      >
        <path d="M0 0l4 5 4-5z" fill="var(--muted)" />
      </svg>
    </div>
  )
}

/**
 * ElevenLabs-style slider: native range input (styled via CSS) + editable value.
 * Click value to type. Uses native <input type="range"> for reliable drag.
 */
export function NodeNumberDrag({
  value,
  onChange,
  min = 0,
  max = 2,
  step = 0.1,
  precision = 1,
}: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  precision?: number
}) {
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const startEdit = useCallback(() => {
    setEditText(value.toFixed(precision))
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 0)
  }, [value, precision])

  const commitEdit = useCallback(() => {
    setEditing(false)
    const parsed = parseFloat(editText)
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, parsed))
      onChange(parseFloat(clamped.toFixed(precision)))
    }
  }, [editText, onChange, min, max, precision])

  return (
    <div className="nodrag" style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%' }}>
      <input
        className="nodrag node-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(parseFloat(e.target.value).toFixed(precision)))}
        style={{ flex: 1, minWidth: 40 }}
      />

      {/* Value display / edit */}
      {editing ? (
        <input
          ref={inputRef}
          className="nodrag"
          type="text"
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitEdit()
            if (e.key === 'Escape') setEditing(false)
          }}
          style={{
            width: 42,
            padding: '1px 4px',
            fontSize: 10,
            background: 'var(--bg)',
            border: '1px solid var(--accent)',
            borderRadius: 3,
            color: 'var(--text)',
            fontFamily: 'var(--mono)',
            outline: 'none',
            textAlign: 'right',
            flexShrink: 0,
          }}
        />
      ) : (
        <span
          onClick={startEdit}
          style={{
            fontSize: 10,
            fontFamily: 'var(--mono)',
            color: 'var(--text)',
            cursor: 'text',
            width: 45,
            textAlign: 'right',
            flexShrink: 0,
            userSelect: 'none',
          }}
        >
          {value.toFixed(precision)}
        </span>
      )}
    </div>
  )
}

/** Styled inline widget textarea */
export function NodeTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className="nodrag"
      style={{
        width: '100%',
        resize: 'none',
        minHeight: 36,
        flex: 1,
        padding: '4px 7px',
        fontSize: 9,
        lineHeight: 1.4,
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 3,
        color: 'var(--text)',
        fontFamily: 'var(--font)',
        outline: 'none',
        ...props.style,
      }}
    />
  )
}
