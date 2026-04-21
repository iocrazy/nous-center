import { useState, useEffect, useMemo } from 'react'
import { Copy, Check } from 'lucide-react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { NODE_DEFS } from '../../models/workflow'
import { useWorkspaceStore } from '../../stores/workspace'
import BaseNode from './BaseNode'

export default function TextOutputNode({ id, data, selected }: NodeProps) {
  const def = NODE_DEFS.text_output
  const [streamText, setStreamText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [copied, setCopied] = useState(false)

  // Find the upstream node that feeds this TextOutput's "text" input.
  // Streaming tokens are emitted with that upstream node's id (e.g. LLM),
  // not ours — we need to listen for the upstream id.
  const tabs = useWorkspaceStore((s) => s.tabs)
  const activeTabId = useWorkspaceStore((s) => s.activeTabId)
  const upstreamNodeId = useMemo(() => {
    const wf = tabs.find((t) => t.id === activeTabId)?.workflow
    const edge = wf?.edges.find((e) => e.target === id)
    return edge?.source ?? null
  }, [tabs, activeTabId, id])

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail
      const streamSource = upstreamNodeId
      if (!streamSource) return
      if (detail.type === 'node_start' && detail.node_id === streamSource) {
        setStreamText('')
        setIsStreaming(true)
      }
      if (detail.type === 'node_stream' && detail.node_id === streamSource) {
        // Wave 1 renamed the stream chunk field from `token` to `content`
        // (workflow_executor dispatch layer). Fall back to `token` for any
        // legacy plugin executor that still emits the old shape.
        const chunk = (detail.content ?? detail.token ?? '') as string
        if (chunk) setStreamText((prev) => prev + chunk)
      }
      if (detail.type === 'node_complete' && detail.node_id === streamSource) {
        setIsStreaming(false)
      }
    }
    window.addEventListener('node-progress', handler)
    return () => window.removeEventListener('node-progress', handler)
  }, [upstreamNodeId])

  // While streaming is active, prefer the rolling streamText so the output
  // node updates in real time. Once streaming ends, fall back to data.text
  // (which the executor updates with the final stripped result).
  const text = isStreaming
    ? streamText
    : ((data.text as string) || streamText || '')

  return (
    <>
    <NodeResizer
      isVisible={selected}
      minWidth={220}
      minHeight={80}
      onResizeEnd={() => window.dispatchEvent(new Event('node-resize-end'))}
      lineStyle={{ borderColor: 'var(--accent)', borderWidth: 1 }}
      handleStyle={{
        width: 10,
        height: 10,
        background: 'var(--accent)',
        border: '2px solid var(--card)',
        borderRadius: 2,
      }}
    />
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px', minHeight: 40, flex: 1, display: 'flex', flexDirection: 'column' }}>
        {text ? (
          <div style={{ position: 'relative', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <button
              className="nodrag"
              onClick={() => {
                navigator.clipboard.writeText(text)
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              }}
              style={{
                position: 'absolute',
                top: 4,
                right: 4,
                background: 'var(--bg-hover)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '3px 5px',
                cursor: 'pointer',
                color: copied ? 'var(--ok)' : 'var(--muted)',
                display: 'flex',
                alignItems: 'center',
                gap: 3,
                fontSize: 9,
                zIndex: 1,
                opacity: 0.7,
                transition: 'opacity 0.15s',
              }}
              onMouseOver={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = '1' }}
              onMouseOut={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = '0.7' }}
            >
              {copied ? <Check size={10} /> : <Copy size={10} />}
              {copied ? '已复制' : '复制'}
            </button>
            <div style={{
              padding: '8px',
              background: 'var(--bg)',
              borderRadius: 4,
              fontSize: 12,
              lineHeight: 1.5,
              flex: 1,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              color: 'var(--text)',
              border: '1px solid var(--border)',
            }}>
              {text}
              {streamText && <span style={{ opacity: 0.5 }}>▍</span>}
            </div>
          </div>
        ) : (
          <div style={{
            flex: 1,
            minHeight: 32,
            background: 'var(--bg)',
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            color: 'var(--muted-strong)',
          }}>
            等待文本输入...
          </div>
        )}
      </div>
    </BaseNode>
    </>
  )
}
