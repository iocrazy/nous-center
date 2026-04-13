import { useRef, useCallback } from 'react'
import { Upload, X, Music } from 'lucide-react'
import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeInput } from './BaseNode'

export default function RefAudioNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.ref_audio
  const dropRef = useRef<HTMLDivElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('audio/')) return
      // Store as base64 data URL for portability
      const reader = new FileReader()
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string
        updateNode(id, { audio_data: dataUrl, path: file.name })
      }
      reader.readAsDataURL(file)
    },
    [id, updateNode],
  )

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--accent-2)'
  }

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--border)'
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--border)'
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  const audioData = (data.audio_data as string) ?? ''
  const fileName = (data.path as string) ?? ''

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'TTS', bg: 'var(--accent-2-subtle)', color: 'var(--accent-2)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      {/* Audio upload / preview */}
      <div style={{ padding: '4px 10px' }}>
        {audioData ? (
          <div style={{
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: 8,
          }}>
            <div className="flex items-center gap-2" style={{ marginBottom: 6 }}>
              <Music size={14} style={{ color: 'var(--accent-2)', flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: 'var(--text)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {fileName}
              </span>
              <button
                className="nodrag"
                onClick={() => updateNode(id, { audio_data: '', path: '' })}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--muted)', padding: 2, display: 'flex', flexShrink: 0,
                }}
              >
                <X size={12} />
              </button>
            </div>
            <audio
              className="nodrag"
              controls
              src={audioData}
              style={{ width: '100%', height: 32 }}
            />
          </div>
        ) : (
          <div
            ref={dropRef}
            className="nodrag"
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => document.getElementById(`audio-file-${id}`)?.click()}
            style={{
              border: '1.5px dashed var(--border)',
              borderRadius: 4,
              padding: '12px 8px',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.15s',
            }}
          >
            <Upload size={16} style={{ color: 'var(--muted)', margin: '0 auto 4px' }} />
            <div style={{ fontSize: 10, color: 'var(--muted)' }}>
              点击或拖拽上传音频
            </div>
            <div style={{ fontSize: 9, color: 'var(--muted-strong)', marginTop: 2 }}>
              WAV / MP3 / FLAC / OGG
            </div>
          </div>
        )}
        <input
          id={`audio-file-${id}`}
          type="file"
          accept="audio/*"
          style={{ display: 'none' }}
          onChange={onInputChange}
        />
      </div>

      {/* Manual path input (fallback) */}
      <NodeWidgetRow label="路径">
        <NodeInput
          value={fileName}
          onChange={(e) => updateNode(id, { path: e.target.value })}
          placeholder="或输入音频路径"
        />
      </NodeWidgetRow>
      <NodeWidgetRow label="参考文本">
        <NodeInput
          value={(data.ref_text as string) ?? ''}
          onChange={(e) => updateNode(id, { ref_text: e.target.value })}
          placeholder="参考文本 (可选)"
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}
