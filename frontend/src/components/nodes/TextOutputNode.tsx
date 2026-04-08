import { useState, useEffect } from 'react'
import type { NodeProps } from '@xyflow/react'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'

export default function TextOutputNode({ id, data, selected }: NodeProps) {
  const def = NODE_DEFS.text_output
  const [streamText, setStreamText] = useState('')

  // Listen for streaming tokens
  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail
      if (detail.type === 'node_stream' && detail.node_id === id) {
        setStreamText((prev) => prev + detail.token)
      }
      if (detail.type === 'node_complete' && detail.node_id === id) {
        setStreamText('')
      }
    }
    window.addEventListener('node-progress', handler)
    return () => window.removeEventListener('node-progress', handler)
  }, [id])

  const text = (data.text as string) || streamText || ''

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px', minHeight: 40 }}>
        {text ? (
          <div style={{
            padding: '8px',
            background: 'var(--bg)',
            borderRadius: 4,
            fontSize: 12,
            lineHeight: 1.5,
            maxHeight: 200,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            color: 'var(--text)',
            border: '1px solid var(--border)',
          }}>
            {text}
            {streamText && <span style={{ opacity: 0.5 }}>▍</span>}
          </div>
        ) : (
          <div style={{
            height: 32,
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
  )
}
