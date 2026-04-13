import { useRef, useCallback, useEffect, useState } from 'react'
import { ImagePlus, X } from 'lucide-react'
import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'

export default function MultimodalInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.multimodal_input
  const dropRef = useRef<HTMLDivElement>(null)

  // images stored as array, backward compat with single image string
  const images: string[] = (() => {
    const raw = data.images ?? data.image
    if (!raw) return []
    if (Array.isArray(raw)) return raw as string[]
    if (typeof raw === 'string' && raw.startsWith('data:')) return [raw]
    return []
  })()

  const setImages = useCallback(
    (imgs: string[]) => updateNode(id, { images: imgs, image: imgs[0] ?? '' }),
    [id, updateNode],
  )

  const addFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string
        setImages([...images, dataUrl])
      }
      reader.readAsDataURL(file)
    },
    [images, setImages],
  )

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      Array.from(files).forEach((f) => {
        if (f.type.startsWith('image/')) addFile(f)
      })
    },
    [addFile],
  )

  const removeImage = useCallback(
    (idx: number) => setImages(images.filter((_, i) => i !== idx)),
    [images, setImages],
  )

  // Paste support (Ctrl+V)
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) addFile(file)
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [addFile])

  // Insert @img tag at cursor
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const insertAtRef = useCallback(
    (idx: number) => {
      const ta = textareaRef.current
      if (!ta) return
      const tag = `@img${idx + 1} `
      const pos = ta.selectionStart ?? ta.value.length
      const newText = ta.value.slice(0, pos) + tag + ta.value.slice(pos)
      updateNode(id, { text: newText })
      setTimeout(() => {
        ta.selectionStart = ta.selectionEnd = pos + tag.length
        ta.focus()
      }, 0)
    },
    [id, updateNode],
  )

  // @ mention popup
  const [showAtPopup, setShowAtPopup] = useState(false)
  const [atPos, setAtPos] = useState({ top: 0, left: 0 })

  const handleTextChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const val = e.target.value
      updateNode(id, { text: val })

      // Check if user just typed @
      const pos = e.target.selectionStart
      if (pos > 0 && val[pos - 1] === '@' && images.length > 0) {
        // Position popup near cursor
        const ta = e.target
        const rect = ta.getBoundingClientRect()
        // Approximate cursor position
        const lines = val.slice(0, pos).split('\n')
        const lineIdx = lines.length - 1
        const charIdx = lines[lineIdx].length
        setAtPos({
          top: rect.top + (lineIdx + 1) * 20 - ta.scrollTop + 4,
          left: rect.left + Math.min(charIdx * 7, rect.width - 100),
        })
        setShowAtPopup(true)
      } else {
        setShowAtPopup(false)
      }
    },
    [id, updateNode, images.length],
  )

  const selectAtImage = useCallback(
    (idx: number) => {
      const ta = textareaRef.current
      if (!ta) return
      const pos = ta.selectionStart
      const val = ta.value
      // Replace the @ with @imgN
      const before = val.slice(0, pos)
      const after = val.slice(pos)
      const tag = `img${idx + 1} `
      updateNode(id, { text: before + tag + after })
      setShowAtPopup(false)
      setTimeout(() => {
        ta.selectionStart = ta.selectionEnd = pos + tag.length
        ta.focus()
      }, 0)
    },
    [id, updateNode],
  )

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
    addFiles(e.dataTransfer.files)
  }

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files)
    e.target.value = ''
  }

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
          ref={textareaRef}
          value={(data.text as string) ?? ''}
          onChange={handleTextChange}
          onBlur={() => setTimeout(() => setShowAtPopup(false), 200)}
          placeholder={images.length > 0 ? '输入文本... 输入 @ 引用图片' : '输入文本...'}
          rows={5}
          className="nodrag nowheel"
          style={{
            width: '100%',
            minHeight: 80,
            padding: 8,
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg)',
            color: 'var(--text)',
            fontSize: 13,
            lineHeight: 1.5,
            resize: 'none',
            fontFamily: 'inherit',
          }}
        />
      </div>

      {/* @ mention popup — outside textarea wrapper to avoid IME issues */}
      {showAtPopup && images.length > 0 && (
        <div
          className="nodrag"
          style={{
            margin: '0 10px 4px',
            background: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            padding: 4,
            display: 'flex',
            gap: 4,
            flexWrap: 'wrap',
          }}
        >
          {images.map((img, idx) => (
            <div
              key={idx}
              onClick={() => selectAtImage(idx)}
              style={{
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '3px 6px',
                borderRadius: 4,
                fontSize: 10,
                color: 'var(--text)',
                background: 'var(--bg)',
                border: '1px solid var(--border)',
              }}
              onMouseOver={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--accent)' }}
              onMouseOut={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border)' }}
            >
              <img src={img} alt="" style={{ width: 24, height: 24, objectFit: 'cover', borderRadius: 3 }} />
              <span>@img{idx + 1}</span>
            </div>
          ))}
        </div>
      )}

      {/* Image thumbnails grid */}
      {images.length > 0 && (
        <div
          style={{
            padding: '0 10px 4px',
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
          }}
        >
          {images.map((img, idx) => (
            <div
              key={idx}
              style={{ position: 'relative', flexShrink: 0 }}
            >
              <img
                src={img}
                alt={`img${idx + 1}`}
                style={{
                  width: 72,
                  height: 72,
                  objectFit: 'cover',
                  borderRadius: 4,
                  border: '1px solid var(--border)',
                  cursor: 'pointer',
                  display: 'block',
                }}
                title={`@img${idx + 1}`}
                onClick={() => insertAtRef(idx)}
              />
              {/* Index badge */}
              <span
                style={{
                  position: 'absolute',
                  bottom: 2,
                  left: 2,
                  fontSize: 8,
                  background: 'rgba(0,0,0,0.7)',
                  color: '#fff',
                  padding: '1px 4px',
                  borderRadius: 3,
                }}
              >
                @img{idx + 1}
              </span>
              {/* Delete button */}
              <button
                className="nodrag"
                onClick={(e) => {
                  e.stopPropagation()
                  removeImage(idx)
                }}
                style={{
                  position: 'absolute',
                  top: 2,
                  right: 2,
                  width: 16,
                  height: 16,
                  borderRadius: '50%',
                  border: 'none',
                  background: 'rgba(0,0,0,0.6)',
                  color: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: 0,
                }}
              >
                <X size={10} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Upload area */}
      <div style={{ padding: '0 10px 8px' }}>
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
            padding: images.length > 0 ? '6px 8px' : '14px 8px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'border-color 0.15s',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
          }}
        >
          <ImagePlus size={14} style={{ color: 'var(--muted)' }} />
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>
            {images.length > 0 ? '继续添加' : '点击或拖拽上传图片'}
          </span>
          {images.length === 0 && (
            <span style={{ fontSize: 9, color: 'var(--muted-strong)' }}>
              也可粘贴（Ctrl+V）
            </span>
          )}
        </div>
        <input
          id={`mm-file-${id}`}
          type="file"
          accept="image/*"
          multiple
          style={{ display: 'none' }}
          onChange={onInputChange}
        />
      </div>
    </BaseNode>
  )
}
