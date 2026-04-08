import { useRef, useCallback, useEffect } from 'react'
import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'

export default function MultimodalInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.multimodal_input
  const dropRef = useRef<HTMLDivElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string
        updateNode(id, { image: dataUrl })
      }
      reader.readAsDataURL(file)
    },
    [id, updateNode],
  )

  // Paste support (Ctrl+V)
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) handleFile(file)
          break
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [handleFile])

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dropRef.current) dropRef.current.style.borderColor = 'var(--accent)'
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

  const imageDataUrl = (data.image as string) ?? ''

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      {/* Text area */}
      <div style={{ padding: '4px 10px' }}>
        <textarea
          value={(data.text as string) ?? ''}
          onChange={(e) => updateNode(id, { text: e.target.value })}
          placeholder="输入文本..."
          rows={5}
          className="nodrag"
          style={{
            width: '100%',
            minHeight: 100,
            padding: 8,
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg)',
            color: 'var(--text)',
            fontSize: 13,
            lineHeight: 1.5,
            resize: 'vertical',
            fontFamily: 'inherit',
          }}
        />
      </div>

      {/* Image upload area */}
      <div style={{ padding: '0 10px 8px' }}>
        {imageDataUrl ? (
          /* Thumbnail preview */
          <div
            style={{
              position: 'relative',
              display: 'inline-block',
              width: '100%',
            }}
          >
            <img
              src={imageDataUrl}
              alt="uploaded"
              style={{
                width: '100%',
                maxHeight: 150,
                objectFit: 'contain',
                borderRadius: 4,
                border: '1px solid var(--border)',
                display: 'block',
              }}
            />
            {/* Delete button */}
            <button
              className="nodrag"
              onClick={() => updateNode(id, { image: '' })}
              title="删除图片"
              style={{
                position: 'absolute',
                top: 4,
                right: 4,
                width: 20,
                height: 20,
                borderRadius: '50%',
                border: 'none',
                background: 'rgba(0,0,0,0.6)',
                color: '#fff',
                fontSize: 12,
                lineHeight: '20px',
                textAlign: 'center',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 0,
              }}
            >
              ×
            </button>
          </div>
        ) : (
          /* Drop zone */
          <div
            ref={dropRef}
            className="nodrag"
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => document.getElementById(`mm-file-${id}`)?.click()}
            style={{
              border: '1.5px dashed var(--border)',
              borderRadius: 4,
              padding: '14px 8px',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'border-color 0.15s',
            }}
          >
            <div style={{ fontSize: 18, color: 'var(--muted)', marginBottom: 4 }}>🖼</div>
            <div style={{ fontSize: 10, color: 'var(--muted)' }}>
              点击或拖拽上传图片
            </div>
            <div style={{ fontSize: 9, color: 'var(--muted-strong)', marginTop: 2 }}>
              也可粘贴（Ctrl+V）
            </div>
          </div>
        )}

        {/* Hidden file input */}
        <input
          id={`mm-file-${id}`}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={onInputChange}
        />
      </div>
    </BaseNode>
  )
}
